//! Owner-consuming production lowering from exact semantic MIR to Kernel IR.
//!
//! This boundary is intentionally fail-closed. It admits only operations with
//! explicit semantic correspondence rules, including trusted invocation
//! capabilities, structured control flow, typed memory access, synchronization,
//! and cooperative matrix operations. The resulting Kernel IR is verified
//! before release. Detached workload markers are not used by this API.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::{error::Error, fmt};

use fe2o3_kernel_ir::{
    AccessMode, AddressSpace, AmdGpuDiagnosticOperation, Atomic, AtomicKind, Axis,
    BarrierSemantics, BasicBlock, BinaryOp, BlockId, CastKind, CheckedBinaryOperator,
    ComparePredicate, Constant, Convergence, F32MathFunction, FloatConversionKind, FloatOperation,
    FormalMemoryIncompleteReason, Function, FunctionBody, FunctionId, FunctionOperationLocation,
    Gfx950LdsTransposeFormatV1, Gfx950LdsTransposeOperationKindV1, Gfx950LdsTransposeOperationV1,
    IndexKind, IntrinsicKind, IntrinsicOperation, Kernel, LaunchDomain, LaunchExtent,
    MAX_OPERATIONS_V1 as MAX_BLOCK_OPERATIONS_V1, MatrixOperation, MatrixOperationKind,
    MemoryAccess, MemoryEffect, MemoryIntrinsicOperation, MemoryOrdering, Module, Operation,
    OperationKind, ScalarType, Signature, SwitchCase, SynchronizationScope, TensorLayoutContractV1,
    Terminator, Type, UnaryOp, ValueDef, ValueId, VerificationErrors,
    VerifiedCanonicalKernelIrErrorV8, VerifiedCanonicalKernelIrErrorV9,
    VerifiedCanonicalKernelIrIdentityV8, VerifiedCanonicalKernelIrIdentityV9,
    VerifiedCanonicalKernelIrV8, VerifiedCanonicalKernelIrV9, WaveF32ReductionKindV1,
    WaveOperation, WaveOperationKind, WaveWidth, WorkgroupBarrier, WorkgroupMemory,
    WorkgroupMemoryExtent, WorkgroupSize, plan_integer_cast_v1, verify_module,
};
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticAbiPassModeV1, SemanticAbiPointeeKindV1, SemanticAggregateKindV1,
    SemanticAssertMessageV1, SemanticAtomicOrderingV1, SemanticAtomicRmwOpV1, SemanticAtomicRmwV1,
    SemanticAtomicScopeV1, SemanticAxisV1, SemanticBackendPrimitiveV1, SemanticBackendReprV1,
    SemanticBf16ConversionKindV1, SemanticBinaryOpV1, SemanticBlockIdV1, SemanticCallableDeclV1,
    SemanticCastKindV1, SemanticCheckedBinaryOpV1, SemanticCompilerIntrinsicOperationV1,
    SemanticConstantValueV1, SemanticDirectCallV1, SemanticDisjointIndexSpaceV1,
    SemanticEnumEncodingV1, SemanticEnumVariantV1, SemanticF32MathFunctionV1,
    SemanticFieldsShapeV1, SemanticFunctionDeclV1, SemanticFunctionIdV1,
    SemanticGfx950LdsTransposeFormatV1, SemanticLocalIdV1, SemanticLocalRoleV1,
    SemanticMfmaAccumulatorContractV1, SemanticMfmaOperandContractV1, SemanticMfmaOperandRoleV1,
    SemanticMfmaProfileV1, SemanticMfmaRegisterDistributionV1, SemanticMfmaStorageLayoutV1,
    SemanticMutabilityV1, SemanticOperandV1, SemanticPlaceV1, SemanticPointerKindV1,
    SemanticPointerMetadataV1, SemanticProjectionKindV1, SemanticRustcVariantsV1,
    SemanticRvalueKindV1, SemanticScalarTypeV1, SemanticScalarValueV1,
    SemanticSourceArgumentOwnershipV1, SemanticStatementKindV1, SemanticSubgroupReductionKindV1,
    SemanticTerminatorKindV1, SemanticTypeDeclV1, SemanticTypeIdV1, SemanticTypeLayoutDetailsV1,
    SemanticTypeShapeV1, SemanticUnaryOpV1, SemanticUncheckedBinaryOpV1, SemanticUnwindActionV1,
    SemanticVolatilityV1, SemanticWorkgroupPipelineEventV1, semantic_direct_enum_variant_v1,
    semantic_scalar_enum_variant_v1,
};
use fe2o3_mir_model::{
    SemanticEnumPayloadDominanceV1, SemanticOptionAvailabilityV1, SemanticOptionDominanceV1,
    semantic_option_producers_v1,
};
use fe2o3_pliron::{
    MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2, PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2,
    ProductionNumericalContractV2, ProductionOverflowContractV2,
    ProductionRankedKernelLoweringInputV1, ProductionRankedOperationV1, ProductionRankedValueIdV1,
    ProductionRankedValueV1, ProductionSemanticBinaryOpV2, ProductionSemanticCastV2,
    ProductionSemanticComparisonV2, ProductionSemanticExpressionV2, ProductionSemanticMirErrorV1,
    ProductionSemanticMirOwnerV1, ProductionSemanticScalarTypeV2, ProductionSemanticUnaryOpV2,
};

const DEFAULT_MAX_FUNCTIONS_V1: usize = 1_024;
const DEFAULT_MAX_BLOCKS_V1: usize = 16_384;
const DEFAULT_MAX_STATEMENTS_V1: usize = 1_048_576;
const DEFAULT_MAX_OPERATIONS_V1: usize = 1_048_576;

/// Independent work limits for semantic-MIR-to-Kernel-IR lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionSemanticKirLimitsV1 {
    max_functions: usize,
    max_blocks: usize,
    max_statements: usize,
    max_operations: usize,
}

impl ProductionSemanticKirLimitsV1 {
    /// Constructs explicit lowering limits.
    pub const fn new(max_functions: usize, max_blocks: usize, max_statements: usize) -> Self {
        Self::new_with_max_operations(
            max_functions,
            max_blocks,
            max_statements,
            DEFAULT_MAX_OPERATIONS_V1,
        )
    }

    /// Constructs explicit lowering limits, including the module-wide emitted-operation budget.
    pub const fn new_with_max_operations(
        max_functions: usize,
        max_blocks: usize,
        max_statements: usize,
        max_operations: usize,
    ) -> Self {
        Self {
            max_functions,
            max_blocks,
            max_statements,
            max_operations,
        }
    }
}

impl Default for ProductionSemanticKirLimitsV1 {
    fn default() -> Self {
        Self::new_with_max_operations(
            DEFAULT_MAX_FUNCTIONS_V1,
            DEFAULT_MAX_BLOCKS_V1,
            DEFAULT_MAX_STATEMENTS_V1,
            DEFAULT_MAX_OPERATIONS_V1,
        )
    }
}

/// A bounded resource charged by production target-neutral lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionSemanticKirResourceV1 {
    /// Semantic functions inspected.
    Functions,
    /// Semantic blocks inspected and materialized.
    Blocks,
    /// Semantic statements inspected.
    Statements,
    /// Kernel IR operations emitted across all blocks.
    Operations,
    /// Sparse exact source-debug bindings retained by lowering.
    DebugBindings,
}

/// Pointer-independent evidence relating one source block to one Kernel IR block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticKirBlockCorrespondenceV1 {
    semantic_function: SemanticFunctionIdV1,
    semantic_block: SemanticBlockIdV1,
    kernel_ir_block: BlockId,
    source_statement_count: u32,
}

/// Exact one-to-one mapping from one selected semantic argument local to a
/// canonical KIR function parameter.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SemanticKirParameterBindingV1 {
    semantic_function: SemanticFunctionIdV1,
    semantic_local: SemanticLocalIdV1,
    kernel_ir_value: ValueId,
}

impl SemanticKirParameterBindingV1 {
    /// Returns the semantic function that owns the parameter local.
    pub const fn semantic_function(self) -> SemanticFunctionIdV1 {
        self.semantic_function
    }

    /// Returns the semantic MIR local represented by this binding.
    pub const fn semantic_local(self) -> SemanticLocalIdV1 {
        self.semantic_local
    }

    /// Returns the exact Kernel IR function-parameter value.
    pub const fn kernel_ir_value(self) -> ValueId {
        self.kernel_ir_value
    }
}

impl SemanticKirBlockCorrespondenceV1 {
    /// Returns the exact semantic function locator.
    pub const fn semantic_function(self) -> SemanticFunctionIdV1 {
        self.semantic_function
    }

    /// Returns the exact semantic block locator.
    pub const fn semantic_block(self) -> SemanticBlockIdV1 {
        self.semantic_block
    }

    /// Returns the corresponding Kernel IR block identity.
    pub const fn kernel_ir_block(self) -> BlockId {
        self.kernel_ir_block
    }

    /// Returns the number of source statements covered by this block rule.
    pub const fn source_statement_count(self) -> u32 {
        self.source_statement_count
    }
}

/// Exact Kernel IR operation span emitted by one semantic MIR statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticKirStatementOperationSpanV1 {
    semantic_function: SemanticFunctionIdV1,
    semantic_block: SemanticBlockIdV1,
    statement_ordinal: u32,
    kernel_ir_block: BlockId,
    first_operation_ordinal: u32,
    operation_count: u32,
}

impl SemanticKirStatementOperationSpanV1 {
    /// Returns the exact semantic function locator.
    pub const fn semantic_function(self) -> SemanticFunctionIdV1 {
        self.semantic_function
    }

    /// Returns the exact semantic block locator.
    pub const fn semantic_block(self) -> SemanticBlockIdV1 {
        self.semantic_block
    }

    /// Returns the zero-based statement ordinal within the semantic block.
    pub const fn statement_ordinal(self) -> u32 {
        self.statement_ordinal
    }

    /// Returns the Kernel IR block that contains the emitted operations.
    pub const fn kernel_ir_block(self) -> BlockId {
        self.kernel_ir_block
    }

    /// Returns the zero-based ordinal of the first emitted operation.
    pub const fn first_operation_ordinal(self) -> u32 {
        self.first_operation_ordinal
    }

    /// Returns the exact number of emitted operations, including zero.
    pub const fn operation_count(self) -> u32 {
        self.operation_count
    }
}

/// Exact Kernel IR operation span emitted while lowering one semantic MIR terminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticKirTerminatorOperationSpanV1 {
    semantic_function: SemanticFunctionIdV1,
    semantic_block: SemanticBlockIdV1,
    kernel_ir_block: BlockId,
    first_operation_ordinal: u32,
    operation_count: u32,
}

impl SemanticKirTerminatorOperationSpanV1 {
    /// Returns the exact semantic function locator.
    pub const fn semantic_function(self) -> SemanticFunctionIdV1 {
        self.semantic_function
    }

    /// Returns the exact semantic block locator.
    pub const fn semantic_block(self) -> SemanticBlockIdV1 {
        self.semantic_block
    }

    /// Returns the Kernel IR block that contains the emitted operations.
    pub const fn kernel_ir_block(self) -> BlockId {
        self.kernel_ir_block
    }

    /// Returns the zero-based ordinal of the first emitted operation.
    pub const fn first_operation_ordinal(self) -> u32 {
        self.first_operation_ordinal
    }

    /// Returns the exact number of operations emitted by the terminator.
    pub const fn operation_count(self) -> u32 {
        self.operation_count
    }
}

/// Closed lowering rule responsible for operations without a semantic MIR source construct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticKirSyntheticOperationRuleV1 {
    /// Private per-invocation slots that preserve enum payloads across discriminant SSA joins.
    EnumPayloadStorage,
    /// The canonical trap operation in the shared runtime-assert failure block.
    RuntimeAssertFailureTrap,
}

/// Exact Kernel IR operation span emitted by one typed synthetic lowering rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticKirSyntheticOperationSpanV1 {
    rule: SemanticKirSyntheticOperationRuleV1,
    kernel_ir_block: BlockId,
    first_operation_ordinal: u32,
    operation_count: u32,
}

impl SemanticKirSyntheticOperationSpanV1 {
    /// Returns the closed synthetic lowering rule.
    pub const fn rule(self) -> SemanticKirSyntheticOperationRuleV1 {
        self.rule
    }

    /// Returns the Kernel IR block that contains the synthetic operations.
    pub const fn kernel_ir_block(self) -> BlockId {
        self.kernel_ir_block
    }

    /// Returns the zero-based ordinal of the first synthetic operation.
    pub const fn first_operation_ordinal(self) -> u32 {
        self.first_operation_ordinal
    }

    /// Returns the exact number of operations emitted by the synthetic rule.
    pub const fn operation_count(self) -> u32 {
        self.operation_count
    }
}

/// Stable operation-attribution trace retained by one live lowering owner.
///
/// Span records identify which lowering invocation emitted each operation. They
/// do not independently prove that an operation implements its source
/// construct. Semantic authority remains with [`ProductionSemanticKirOwnerV1`],
/// whose equivalence check replays lowering and compares the complete module
/// and trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticKirCorrespondenceV1 {
    semantic_sha256: [u8; 32],
    function_count: usize,
    blocks: Box<[SemanticKirBlockCorrespondenceV1]>,
    statement_operation_spans: Box<[SemanticKirStatementOperationSpanV1]>,
    terminator_operation_spans: Box<[SemanticKirTerminatorOperationSpanV1]>,
    synthetic_operation_spans: Box<[SemanticKirSyntheticOperationSpanV1]>,
    parameter_bindings: Box<[SemanticKirParameterBindingV1]>,
}

impl SemanticKirCorrespondenceV1 {
    /// Returns the exact admitted semantic identity.
    pub const fn semantic_sha256(&self) -> &[u8; 32] {
        &self.semantic_sha256
    }

    /// Returns the number of functions retained by the admitted semantic MIR.
    ///
    /// For a `KernelResult` entry this includes both the exact unit-ABI wrapper
    /// and its ordinary Rust body, even though only the selected body produces
    /// Kernel IR operations.
    pub const fn function_count(&self) -> usize {
        self.function_count
    }

    /// Returns source-to-Kernel-IR block evidence in lowering order.
    pub fn blocks(&self) -> &[SemanticKirBlockCorrespondenceV1] {
        &self.blocks
    }

    /// Returns exact source-statement operation spans in lowering order.
    ///
    /// Zero-operation statements are represented by an explicit zero-length
    /// span at the current operation ordinal.
    pub fn statement_operation_spans(&self) -> &[SemanticKirStatementOperationSpanV1] {
        &self.statement_operation_spans
    }

    /// Returns exact source-terminator operation spans in lowering order.
    pub fn terminator_operation_spans(&self) -> &[SemanticKirTerminatorOperationSpanV1] {
        &self.terminator_operation_spans
    }

    /// Returns operation spans introduced by closed synthetic lowering rules.
    pub fn synthetic_operation_spans(&self) -> &[SemanticKirSyntheticOperationSpanV1] {
        &self.synthetic_operation_spans
    }

    /// Returns sparse exact argument-local to KIR-parameter correspondence.
    pub fn parameter_bindings(&self) -> &[SemanticKirParameterBindingV1] {
        &self.parameter_bindings
    }

    fn validate_layout_against(
        &self,
        semantic_owner: &ProductionSemanticMirOwnerV1,
        module: &Module,
        has_runtime_assert: bool,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        validate_semantic_kir_correspondence(semantic_owner, module, self, has_runtime_assert)
    }
}

/// Fail-closed diagnostic from independent MIR-to-PLIRON translation
/// validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionMirPlironTranslationErrorV1 {
    /// The retained graphs exceed the independent validation work budget.
    ResourceLimit,
    /// The direct Kernel IR projection does not contain one selected kernel.
    KernelShape,
    /// A non-private executable effect has no exact source attribution.
    UnattributedExecutableEffect {
        /// Exact Kernel IR operation location.
        location: FunctionOperationLocation,
    },
    /// One source-attributed executable effect has no ranked counterpart.
    MissingRankedEffect {
        /// Source semantic block.
        semantic_block: u32,
        /// Source statement, or `None` for a terminator.
        semantic_statement: Option<u32>,
        /// Effect ordinal within the source site.
        semantic_access_ordinal: u32,
    },
    /// One ranked effect has no executable counterpart.
    ExtraRankedEffect {
        /// Ranked block.
        ranked_block: u32,
        /// Ranked operation ordinal.
        ranked_operation: u32,
    },
    /// The executable and ranked access kinds differ.
    AccessKindMismatch {
        /// Exact Kernel IR operation location.
        location: FunctionOperationLocation,
    },
    /// Atomic ordering, scope, or failure ordering differs.
    AtomicContractMismatch {
        /// Exact Kernel IR operation location.
        location: FunctionOperationLocation,
    },
    /// The executable and ranked memory spaces differ.
    MemorySpaceMismatch {
        /// Exact Kernel IR operation location.
        location: FunctionOperationLocation,
    },
    /// A global ranked access names a different external allocation.
    AllocationOriginMismatch {
        /// Exact Kernel IR operation location.
        location: FunctionOperationLocation,
    },
    /// The two projections disagree about effect reachability or loop order.
    ControlFlowMismatch {
        /// First exact source effect.
        first_semantic_block: u32,
        /// First source statement, or `None` for a terminator.
        first_semantic_statement: Option<u32>,
        /// Second exact source effect.
        second_semantic_block: u32,
        /// Second source statement, or `None` for a terminator.
        second_semantic_statement: Option<u32>,
    },
    /// A value-carrying ranked write is not the executable write expression.
    ValueExpressionMismatch {
        /// Exact Kernel IR write location.
        location: FunctionOperationLocation,
    },
    /// Synchronization contracts differ between the two projections.
    SynchronizationMismatch,
    /// Cooperative tensor contracts differ between the two projections.
    TensorContractMismatch,
}

impl fmt::Display for ProductionMirPlironTranslationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceLimit => formatter.write_str(
                "MIR-to-PLIRON translation validation exceeded its independent work budget",
            ),
            Self::KernelShape => formatter.write_str(
                "MIR-to-PLIRON translation validation requires one exact executable kernel body",
            ),
            Self::UnattributedExecutableEffect { location } => write!(
                formatter,
                "executable effect at {location:?} has no exact semantic MIR attribution",
            ),
            Self::MissingRankedEffect {
                semantic_block,
                semantic_statement,
                semantic_access_ordinal,
            } => write!(
                formatter,
                "semantic MIR effect <block={semantic_block}, statement={semantic_statement:?}, ordinal={semantic_access_ordinal}> has no ranked PLIRON counterpart",
            ),
            Self::ExtraRankedEffect {
                ranked_block,
                ranked_operation,
            } => write!(
                formatter,
                "ranked PLIRON effect <block={ranked_block}, operation={ranked_operation}> has no executable semantic MIR counterpart",
            ),
            Self::AccessKindMismatch { location } => write!(
                formatter,
                "ranked PLIRON access kind differs from executable semantic MIR at {location:?}",
            ),
            Self::AtomicContractMismatch { location } => write!(
                formatter,
                "ranked PLIRON atomic ordering, scope, or failure ordering differs from executable semantic MIR at {location:?}",
            ),
            Self::MemorySpaceMismatch { location } => write!(
                formatter,
                "ranked PLIRON memory space differs from executable semantic MIR at {location:?}",
            ),
            Self::AllocationOriginMismatch { location } => write!(
                formatter,
                "ranked PLIRON allocation origin differs from executable semantic MIR at {location:?}",
            ),
            Self::ControlFlowMismatch {
                first_semantic_block,
                first_semantic_statement,
                second_semantic_block,
                second_semantic_statement,
            } => write!(
                formatter,
                "ranked PLIRON effect control flow differs between semantic MIR sites <block={first_semantic_block}, statement={first_semantic_statement:?}> and <block={second_semantic_block}, statement={second_semantic_statement:?}>",
            ),
            Self::ValueExpressionMismatch { location } => write!(
                formatter,
                "ranked PLIRON write expression differs from executable semantic MIR at {location:?}",
            ),
            Self::SynchronizationMismatch => formatter.write_str(
                "ranked PLIRON synchronization contracts differ from executable semantic MIR",
            ),
            Self::TensorContractMismatch => formatter.write_str(
                "ranked PLIRON cooperative tensor contracts differ from executable semantic MIR",
            ),
        }
    }
}

impl Error for ProductionMirPlironTranslationErrorV1 {}

/// Fail-closed diagnostics from production target-neutral lowering.
#[derive(Debug)]
pub enum ProductionSemanticKirErrorV1 {
    /// The exact semantic owner failed recursive verification.
    SemanticOwner(ProductionSemanticMirErrorV1),
    /// A bounded lowering resource exceeded its limit.
    ResourceLimit {
        /// Resource that exceeded its limit.
        resource: ProductionSemanticKirResourceV1,
        /// Observed work.
        actual: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Storage for a bounded lowering resource could not be reserved.
    AllocationFailure {
        /// Resource whose bounded storage reservation failed.
        resource: ProductionSemanticKirResourceV1,
    },
    /// A semantic construct has no exact lowering rule.
    Unsupported {
        /// Source semantic function index.
        function: u32,
        /// Source semantic block index when available.
        block: Option<u32>,
        /// Source semantic statement ordinal when available.
        statement: Option<u32>,
        /// Stable rejection reason.
        detail: &'static str,
    },
    /// A semantic local is used before an SSA value is available on this path.
    MissingLocalDefinition {
        /// Source semantic function index.
        function: u32,
        /// Source semantic block index.
        block: u32,
        /// Source semantic statement ordinal when available.
        statement: Option<u32>,
        /// Missing semantic local index.
        local: u32,
    },
    /// A variant-refined enum reached a payload projection without retained fields.
    EnumPayloadUnavailable {
        /// Source semantic function index.
        function: u32,
        /// Source semantic block index.
        block: u32,
        /// Source semantic statement ordinal when available.
        statement: Option<u32>,
        /// Enum carrier local.
        local: u32,
        /// Variant selected by the preceding downcast.
        variant: u32,
        /// Requested payload field.
        field: u32,
        /// Payload fields retained by the current SSA binding.
        available_fields: usize,
        /// Bounded semantic definitions and discriminator uses for the carrier.
        evidence: Vec<String>,
    },
    /// A compiler-issued capability reached a trusted intrinsic as ordinary data.
    CapabilityUnavailable {
        /// Source semantic function index.
        function: u32,
        /// Source semantic block index.
        block: u32,
        /// Trusted operation that requires the capability.
        operation: &'static str,
        /// Binding class observed by the lowerer.
        actual: &'static str,
        /// Source operand local when it is an exact place.
        local: Option<u32>,
        /// Bounded semantic definitions for the operand carrier.
        evidence: Vec<String>,
    },
    /// A semantic place projection cannot be applied to its path binding.
    PlaceProjectionUnavailable {
        /// Source semantic function index.
        function: u32,
        /// Source semantic block index.
        block: u32,
        /// Source semantic statement ordinal when available.
        statement: Option<u32>,
        /// Root place local.
        local: u32,
        /// Binding class before the rejected projection.
        binding: &'static str,
        /// Rejected typed projection.
        projection: String,
        /// Bounded semantic definitions for the root carrier.
        evidence: Vec<String>,
    },
    /// A semantic type requested scalar lowering without a scalar shape.
    ScalarTypeUnavailable {
        /// Requested semantic type index.
        semantic_type: u32,
        /// Exact retained semantic type shape.
        shape: String,
    },
    /// The constructed Kernel IR failed structural or semantic verification.
    InvalidKernelIr(VerificationErrors),
    /// The lowered module could not become exact verified canonical Kernel IR V8.
    CanonicalKernelIrV8(VerifiedCanonicalKernelIrErrorV8),
    /// The lowered collective or LDS transpose module could not become exact verified canonical
    /// Kernel IR V9.
    CanonicalKernelIrV9(VerifiedCanonicalKernelIrErrorV9),
    /// Independent semantic MIR to ranked PLIRON translation validation failed.
    MirPlironTranslation(ProductionMirPlironTranslationErrorV1),
    /// Retained correspondence no longer matches the exact source owner.
    CorrespondenceMismatch,
}

impl fmt::Display for ProductionSemanticKirErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticOwner(error) => write!(formatter, "exact semantic owner failed: {error}"),
            Self::ResourceLimit {
                resource,
                actual,
                limit,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR {resource:?} work {actual} exceeds limit {limit}",
            ),
            Self::AllocationFailure { resource } => write!(
                formatter,
                "semantic-to-Kernel-IR could not reserve bounded {resource:?} storage",
            ),
            Self::Unsupported {
                function,
                block,
                statement,
                detail,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR lowering rejected function {function}, block {block:?}, statement {statement:?}: {detail}",
            ),
            Self::MissingLocalDefinition {
                function,
                block,
                statement,
                local,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR lowering rejected function {function}, block {block}, statement {statement:?}: local {local} has no path-available SSA definition",
            ),
            Self::EnumPayloadUnavailable {
                function,
                block,
                statement,
                local,
                variant,
                field,
                available_fields,
                evidence,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR lowering rejected function {function}, block {block}, statement {statement:?}: enum local {local} variant {variant} requests payload field {field}, but its control-flow SSA binding retains {available_fields} field(s); carrier evidence: {evidence:?}",
            ),
            Self::CapabilityUnavailable {
                function,
                block,
                operation,
                actual,
                local,
                evidence,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR lowering rejected function {function}, block {block}: {operation} requires compiler-issued capability authority, but operand local {local:?} reached KIR as {actual}; carrier evidence: {evidence:?}",
            ),
            Self::PlaceProjectionUnavailable {
                function,
                block,
                statement,
                local,
                binding,
                projection,
                evidence,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR lowering rejected function {function}, block {block}, statement {statement:?}: local {local} binding {binding} does not admit projection {projection}; carrier evidence: {evidence:?}",
            ),
            Self::ScalarTypeUnavailable {
                semantic_type,
                shape,
            } => write!(
                formatter,
                "semantic-to-Kernel-IR scalar lowering rejected semantic type {semantic_type} with shape {shape}",
            ),
            Self::InvalidKernelIr(error) => error.fmt(formatter),
            Self::CanonicalKernelIrV8(error) => {
                write!(
                    formatter,
                    "canonical Kernel IR V8 admission failed: {error}"
                )
            }
            Self::CanonicalKernelIrV9(error) => {
                write!(
                    formatter,
                    "canonical Kernel IR V9 admission failed: {error}"
                )
            }
            Self::MirPlironTranslation(error) => {
                write!(
                    formatter,
                    "MIR-to-PLIRON translation validation failed: {error}"
                )
            }
            Self::CorrespondenceMismatch => formatter.write_str(
                "semantic-to-Kernel-IR correspondence no longer matches its exact owner",
            ),
        }
    }
}

impl Error for ProductionSemanticKirErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SemanticOwner(error) => Some(error),
            Self::InvalidKernelIr(error) => Some(error),
            Self::CanonicalKernelIrV8(error) => Some(error),
            Self::CanonicalKernelIrV9(error) => Some(error),
            Self::MirPlironTranslation(error) => Some(error),
            Self::ResourceLimit { .. }
            | Self::AllocationFailure { .. }
            | Self::Unsupported { .. }
            | Self::MissingLocalDefinition { .. }
            | Self::EnumPayloadUnavailable { .. }
            | Self::CapabilityUnavailable { .. }
            | Self::PlaceProjectionUnavailable { .. }
            | Self::ScalarTypeUnavailable { .. }
            | Self::CorrespondenceMismatch => None,
        }
    }
}

/// Exact source and ranked-graph location of one projected memory access.
///
/// This is compiler-internal correspondence, not proof authority. It is
/// revalidated against all three retained IR owners before formal admission.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProductionRankedAccessSourceV1 {
    semantic_block: u32,
    semantic_statement: Option<u32>,
    semantic_access_ordinal: u32,
    ranked_block: u32,
    ranked_operation: u32,
}

impl ProductionRankedAccessSourceV1 {
    /// Constructs one compiler-projected access correspondence record.
    #[doc(hidden)]
    pub const fn new(
        semantic_block: u32,
        semantic_statement: Option<u32>,
        semantic_access_ordinal: u32,
        ranked_block: u32,
        ranked_operation: u32,
    ) -> Self {
        Self {
            semantic_block,
            semantic_statement,
            semantic_access_ordinal,
            ranked_block,
            ranked_operation,
        }
    }

    /// Returns the exact source semantic block.
    pub const fn semantic_block(self) -> u32 {
        self.semantic_block
    }

    /// Returns the source statement, or `None` for a terminator effect.
    pub const fn semantic_statement(self) -> Option<u32> {
        self.semantic_statement
    }

    /// Returns the access ordinal within the source statement or terminator.
    pub const fn semantic_access_ordinal(self) -> u32 {
        self.semantic_access_ordinal
    }

    /// Returns the block containing the ranked PLIRON access.
    pub const fn ranked_block(self) -> u32 {
        self.ranked_block
    }

    /// Returns the ranked PLIRON operation ordinal.
    pub const fn ranked_operation(self) -> u32 {
        self.ranked_operation
    }
}

/// Move-only custody for one compiler-projected semantic-to-ranked candidate.
///
/// This receipt prevents safe callers from mixing independent semantic,
/// ranked-graph, and diagnostic-IR values. It is not translation validation.
/// Production must independently lower the retained MIR and construct a
/// [`ProductionMirPlironTranslationValidationV1`] before this candidate can
/// enter target-neutral lowering custody.
#[must_use = "dropping the receipt abandons the semantic-to-ranked candidate"]
pub struct ProductionRankedSemanticProjectionReceiptV1 {
    semantic: ProductionSemanticMirOwnerV1,
    lowering: ProductionRankedKernelLoweringInputV1,
    ranked_ir: String,
    semantic_sha256: [u8; 32],
    function_name: String,
    access_sources: Box<[ProductionRankedAccessSourceV1]>,
}

impl fmt::Debug for ProductionRankedSemanticProjectionReceiptV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionRankedSemanticProjectionReceiptV1")
            .field("function_name", &self.function_name)
            .field("ranked_ir_bytes", &self.ranked_ir.len())
            .field("access_sources", &self.access_sources.len())
            .finish_non_exhaustive()
    }
}

/// Borrows one exact semantic root and ranked candidate to validate the
/// structural conditions required by projection-receipt custody.
///
/// Success grants no receipt, translation-validation, Kernel IR, artifact, or
/// launch authority. The semantic owner and ranked lowering remain with the
/// caller so a module-wide owner can validate every member of an ordered
/// ranked roster without cloning or splitting custody.
pub fn validate_borrowed_ranked_semantic_projection_candidate_v1(
    semantic: &ProductionSemanticMirOwnerV1,
    selected_root: SemanticFunctionIdV1,
    lowering: &ProductionRankedKernelLoweringInputV1,
    ranked_ir: &str,
    access_sources: &[ProductionRankedAccessSourceV1],
) -> Result<(), ProductionSemanticKirErrorV1> {
    semantic
        .verify_equivalence()
        .map_err(ProductionSemanticKirErrorV1::SemanticOwner)?;
    if !mandatory_generic_checks_are_clean(lowering) {
        return Err(unsupported(
            0,
            None,
            None,
            "ranked projection receipt contains a rejected mandatory kernel check",
        ));
    }
    if ranked_ir.is_empty() {
        return Err(unsupported(
            0,
            None,
            None,
            "ranked projection receipt has empty diagnostic IR",
        ));
    }
    if !ranked_access_sources_are_well_formed(lowering, access_sources) {
        return Err(unsupported(
            0,
            None,
            None,
            "ranked projection receipt has invalid access correspondence",
        ));
    }
    let document = semantic.semantic();
    let root = document
        .roots()
        .binary_search(&selected_root)
        .ok()
        .and_then(|index| document.roots().get(index))
        .and_then(|root| document.functions().get(root.index() as usize))
        .and_then(SemanticFunctionDeclV1::kernel_entry)
        .ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "ranked projection receipt has no exact kernel root",
            )
        })?;
    let function_name = std::str::from_utf8(root.export_symbol().as_bytes()).map_err(|_| {
        unsupported(
            0,
            None,
            None,
            "ranked projection receipt has a non-UTF-8 kernel symbol",
        )
    })?;
    if function_name != lowering.kernel().function_name() {
        return Err(unsupported(
            0,
            None,
            None,
            "ranked projection receipt function identity changed",
        ));
    }
    Ok(())
}

impl ProductionRankedSemanticProjectionReceiptV1 {
    /// Packages the result of the compiler's deterministic semantic projector.
    ///
    /// This verifies structural custody only. The independently checked
    /// translation relation is constructed later by
    /// [`ProductionSemanticKirOwnerV1::try_lower_after_ranked_checks`].
    #[doc(hidden)]
    pub fn from_unvalidated_projection_candidate(
        semantic: ProductionSemanticMirOwnerV1,
        lowering: ProductionRankedKernelLoweringInputV1,
        ranked_ir: String,
        access_sources: Vec<ProductionRankedAccessSourceV1>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let document = semantic.semantic();
        let [selected_root] = document.roots() else {
            return Err(unsupported(
                0,
                None,
                None,
                "ranked projection receipt has no exact kernel root",
            ));
        };
        validate_borrowed_ranked_semantic_projection_candidate_v1(
            &semantic,
            *selected_root,
            &lowering,
            &ranked_ir,
            &access_sources,
        )?;
        let semantic_sha256 = *document.semantic_sha256().as_bytes();
        let function_name = lowering.kernel().function_name().to_owned();
        Ok(Self {
            semantic_sha256,
            function_name,
            semantic,
            lowering,
            ranked_ir,
            access_sources: access_sources.into_boxed_slice(),
        })
    }

    /// Borrows the exact semantic owner retained by this receipt.
    pub const fn semantic(&self) -> &ProductionSemanticMirOwnerV1 {
        &self.semantic
    }

    /// Borrows the owner-held ranked graph and mandatory check reports.
    pub const fn lowering(&self) -> &ProductionRankedKernelLoweringInputV1 {
        &self.lowering
    }

    /// Borrows the bounded diagnostic ranked IR emitted by the projector.
    pub fn ranked_ir(&self) -> &str {
        &self.ranked_ir
    }

    /// A projection receipt is custody only, never artifact or launch authority.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

/// Independently checked relation between retained semantic MIR effects and
/// the ranked PLIRON program admitted by the mandatory analysis pipeline.
///
/// Construction uses the direct MIR-to-Kernel-IR lowerer as an independent
/// semantic projection. It requires every source-attributed executable read,
/// write, or atomic effect to correspond bijectively to one ranked effect with
/// the same access kind, memory space, external allocation provenance, and
/// atomic ordering/scope contract. Synchronization and cooperative tensor
/// layout contracts are reconciled independently as exact multisets. Memory
/// effect reachability and every represented scalar write expression are also
/// independently reconciled. Volatile, unrepresentable-address-space, or
/// otherwise unattributed effects fail closed.
#[derive(Debug, Eq, PartialEq)]
pub struct ProductionMirPlironTranslationValidationV1 {
    semantic_sha256: [u8; 32],
    memory_effects: usize,
    synchronization_effects: usize,
    tensor_operations: usize,
    value_expressions: usize,
    conservative_ranked_effects: usize,
}

impl ProductionMirPlironTranslationValidationV1 {
    /// Returns the exact admitted semantic MIR identity.
    pub const fn semantic_sha256(&self) -> &[u8; 32] {
        &self.semantic_sha256
    }

    /// Returns bijectively matched executable read, write, and atomic effects.
    pub const fn memory_effects(&self) -> usize {
        self.memory_effects
    }

    /// Returns the number of exactly reconciled synchronization operations.
    pub const fn synchronization_effects(&self) -> usize {
        self.synchronization_effects
    }

    /// Returns the number of exactly reconciled cooperative tensor operations.
    pub const fn tensor_operations(&self) -> usize {
        self.tensor_operations
    }

    /// Returns the number of independently reconstructed scalar write roots.
    pub const fn value_expressions(&self) -> usize {
        self.value_expressions
    }

    /// The compiler projector is not a trusted premise for the effect
    /// occurrence, access kind, memory space, external allocation,
    /// atomic-contract, synchronization-contract, tensor-layout,
    /// memory-effect-flow, and represented scalar-value claims reconciled by
    /// this report.
    ///
    /// This does not apply to ranked address expressions or other claims that
    /// this first validation milestone explicitly leaves outside its scope.
    pub const fn reconciled_projection_remains_trusted(&self) -> bool {
        false
    }

    /// PLIRON does not yet retain a complete physical layout and stride model,
    /// so this report does not claim that every ranked index denotes the same
    /// byte address as Kernel IR.
    pub const fn claims_indexed_address_equivalence(&self) -> bool {
        false
    }

    /// Ranked PLIRON is a verification abstraction, not an executable IR.
    /// This report deliberately makes no claim of whole-program operational
    /// equivalence for source computations that PLIRON does not represent.
    pub const fn claims_complete_operational_equivalence(&self) -> bool {
        false
    }

    /// Returns conservative allocation-level PLIRON effects. These may
    /// over-approximate source behavior but can never erase an executable
    /// source effect.
    pub const fn conservative_ranked_effects(&self) -> usize {
        self.conservative_ranked_effects
    }

    /// This report is compiler correctness evidence only. It grants no object,
    /// publication, load, launch, runtime, or hardware authority.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

/// Move-only owner of one exact semantic source and its verified Kernel IR.
#[must_use = "dropping the owner abandons the verified target-neutral lowering"]
pub struct ProductionSemanticKirOwnerV1 {
    semantic: ProductionSemanticMirOwnerV1,
    module: Module,
    canonical_kernel_ir: ProductionCanonicalKernelIrV1,
    correspondence: SemanticKirCorrespondenceV1,
    limits: ProductionSemanticKirLimitsV1,
    launch_rank: Option<u8>,
    generic_checks: Option<RetainedGenericKernelChecksV1>,
}

#[derive(Debug, Eq, PartialEq)]
enum ProductionCanonicalKernelIrV1 {
    V8(VerifiedCanonicalKernelIrV8),
    V9(VerifiedCanonicalKernelIrV9),
}

/// Exact canonical Kernel IR wire version retained by production lowering.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProductionCanonicalKernelIrVersionV1 {
    /// Exact canonical Kernel IR V8.
    V8,
    /// Exact canonical Kernel IR V9.
    V9,
}

/// Version-bound identity of the canonical Kernel IR bytes retained by the owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionCanonicalKernelIrIdentityV1 {
    version: ProductionCanonicalKernelIrVersionV1,
    digest: [u8; 32],
    canonical_length: u64,
}

impl ProductionCanonicalKernelIrIdentityV1 {
    pub(crate) const fn from_canonical_parts(
        version: ProductionCanonicalKernelIrVersionV1,
        digest: [u8; 32],
        canonical_length: u64,
    ) -> Self {
        Self {
            version,
            digest,
            canonical_length,
        }
    }

    /// Returns the exact canonical wire version committed by this identity.
    pub const fn version(&self) -> ProductionCanonicalKernelIrVersionV1 {
        self.version
    }

    /// Returns the version-domain-separated SHA-256 digest.
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Returns the exact retained canonical byte length.
    pub const fn canonical_length(&self) -> u64 {
        self.canonical_length
    }
}

impl ProductionCanonicalKernelIrV1 {
    fn from_module(module: Module) -> Result<Self, ProductionSemanticKirErrorV1> {
        if module_requires_kernel_ir_v9_collective_or_lds_transpose_v1(&module) {
            VerifiedCanonicalKernelIrV9::from_module(module)
                .map(Self::V9)
                .map_err(ProductionSemanticKirErrorV1::CanonicalKernelIrV9)
        } else {
            VerifiedCanonicalKernelIrV8::from_module(module)
                .map(Self::V8)
                .map_err(ProductionSemanticKirErrorV1::CanonicalKernelIrV8)
        }
    }

    fn revalidate(&self) -> Result<(), ProductionSemanticKirErrorV1> {
        match self {
            Self::V8(owner) => owner
                .revalidate()
                .map_err(ProductionSemanticKirErrorV1::CanonicalKernelIrV8),
            Self::V9(owner) => owner
                .revalidate()
                .map_err(ProductionSemanticKirErrorV1::CanonicalKernelIrV9),
        }
    }

    fn canonical_bytes(&self) -> &[u8] {
        match self {
            Self::V8(owner) => owner.canonical_bytes(),
            Self::V9(owner) => owner.canonical_bytes(),
        }
    }

    fn identity(&self) -> ProductionCanonicalKernelIrIdentityV1 {
        match self {
            Self::V8(owner) => ProductionCanonicalKernelIrIdentityV1 {
                version: ProductionCanonicalKernelIrVersionV1::V8,
                digest: *owner.identity().digest(),
                canonical_length: owner.identity().canonical_length(),
            },
            Self::V9(owner) => ProductionCanonicalKernelIrIdentityV1 {
                version: ProductionCanonicalKernelIrVersionV1::V9,
                digest: *owner.identity().digest(),
                canonical_length: owner.identity().canonical_length(),
            },
        }
    }
}

fn module_requires_kernel_ir_v9_collective_or_lds_transpose_v1(module: &Module) -> bool {
    module.functions.iter().any(|function| {
        function.body.as_ref().is_some_and(|body| {
            body.blocks.iter().any(|block| {
                block.operations.iter().any(|operation| {
                    matches!(
                        operation.kind,
                        OperationKind::Gfx950LdsTranspose(_)
                            | OperationKind::Wave(WaveOperation {
                                kind: WaveOperationKind::ReduceF32 { .. }
                                    | WaveOperationKind::BroadcastF32 { .. },
                                ..
                            })
                    )
                })
            })
        })
    })
}

struct RetainedGenericKernelChecksV1 {
    semantic_sha256: [u8; 32],
    function_name: String,
    ranked_ir: Box<str>,
    lowering: ProductionRankedKernelLoweringInputV1,
    access_sources: Box<[ProductionRankedAccessSourceV1]>,
    translation_validation: ProductionMirPlironTranslationValidationV1,
}

impl fmt::Debug for ProductionSemanticKirOwnerV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionSemanticKirOwnerV1")
            .field("module", &self.module.id)
            .field("canonical_kernel_ir", &self.canonical_kernel_ir)
            .field("correspondence", &self.correspondence)
            .field("limits", &self.limits)
            .field("launch_rank", &self.launch_rank)
            .field("retains_generic_checks", &self.generic_checks.is_some())
            .finish_non_exhaustive()
    }
}

impl ProductionSemanticKirOwnerV1 {
    /// Consumes exact semantic ownership and constructs verified Kernel IR.
    pub fn try_lower(
        semantic: ProductionSemanticMirOwnerV1,
        limits: ProductionSemanticKirLimitsV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        semantic
            .verify_equivalence()
            .map_err(ProductionSemanticKirErrorV1::SemanticOwner)?;
        let (module, correspondence) = lower_module(&semantic, limits, None)?;
        let canonical_kernel_ir = ProductionCanonicalKernelIrV1::from_module(module.clone())?;
        let owner = Self {
            semantic,
            module,
            canonical_kernel_ir,
            correspondence,
            limits,
            launch_rank: None,
            generic_checks: None,
        };
        owner.verify_equivalence()?;
        Ok(owner)
    }

    /// Constructs Kernel IR while retaining the exact ranked graph and every
    /// mandatory generic-check report that admitted the same semantic owner.
    pub fn try_lower_after_ranked_checks(
        receipt: ProductionRankedSemanticProjectionReceiptV1,
        limits: ProductionSemanticKirLimitsV1,
        launch_rank: u8,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let ProductionRankedSemanticProjectionReceiptV1 {
            semantic,
            lowering,
            ranked_ir,
            semantic_sha256,
            function_name,
            access_sources,
        } = receipt;
        semantic
            .verify_equivalence()
            .map_err(ProductionSemanticKirErrorV1::SemanticOwner)?;
        if !mandatory_generic_checks_are_clean(&lowering) {
            return Err(unsupported(
                0,
                None,
                None,
                "ranked proof custody contains a rejected mandatory kernel check",
            ));
        }
        let (module, correspondence) = lower_module(&semantic, limits, Some(launch_rank))?;
        let translation_validation = validate_mir_pliron_translation_v1(
            &module,
            &correspondence,
            &lowering,
            &access_sources,
            limits.max_operations,
        )
        .map_err(ProductionSemanticKirErrorV1::MirPlironTranslation)?;
        let canonical_kernel_ir = ProductionCanonicalKernelIrV1::from_module(module.clone())?;
        let owner = Self {
            semantic,
            module,
            canonical_kernel_ir,
            correspondence,
            limits,
            launch_rank: Some(launch_rank),
            generic_checks: Some(RetainedGenericKernelChecksV1 {
                semantic_sha256,
                function_name,
                ranked_ir: ranked_ir.into_boxed_str(),
                lowering,
                access_sources,
                translation_validation,
            }),
        };
        owner.verify_equivalence()?;
        Ok(owner)
    }

    /// Re-verifies semantic ownership, Kernel IR, and retained correspondence.
    pub fn verify_equivalence(&self) -> Result<(), ProductionSemanticKirErrorV1> {
        self.semantic
            .verify_equivalence()
            .map_err(ProductionSemanticKirErrorV1::SemanticOwner)?;
        self.canonical_kernel_ir.revalidate()?;
        verify_module(&self.module).map_err(ProductionSemanticKirErrorV1::InvalidKernelIr)?;
        let (rederived_module, rederived_correspondence) =
            lower_module(&self.semantic, self.limits, self.launch_rank)?;
        let rederived_canonical_kernel_ir =
            ProductionCanonicalKernelIrV1::from_module(rederived_module.clone())?;
        if self.module != rederived_module
            || self.correspondence != rederived_correspondence
            || self.canonical_kernel_ir != rederived_canonical_kernel_ir
            || self.canonical_kernel_ir.canonical_bytes()
                != rederived_canonical_kernel_ir.canonical_bytes()
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        if let Some(generic_checks) = &self.generic_checks {
            let Some(function) = self.module.functions.first() else {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            };
            if generic_checks.semantic_sha256 != self.correspondence.semantic_sha256
                || generic_checks.function_name != function.id.as_str()
                || generic_checks.ranked_ir.is_empty()
                || !mandatory_generic_checks_are_clean(&generic_checks.lowering)
                || !ranked_access_sources_are_well_formed(
                    &generic_checks.lowering,
                    &generic_checks.access_sources,
                )
            {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            let revalidated = validate_mir_pliron_translation_v1(
                &self.module,
                &self.correspondence,
                &generic_checks.lowering,
                &generic_checks.access_sources,
                self.limits.max_operations,
            )
            .map_err(ProductionSemanticKirErrorV1::MirPlironTranslation)?;
            if revalidated != generic_checks.translation_validation {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        }
        Ok(())
    }

    /// Borrows the retained exact semantic owner.
    pub const fn semantic(&self) -> &ProductionSemanticMirOwnerV1 {
        &self.semantic
    }

    /// Borrows the structurally verified Kernel IR module.
    pub const fn module(&self) -> &Module {
        &self.module
    }

    /// Borrows the authoritative exact, semantically verified Kernel IR V8 bytes.
    pub const fn canonical_kernel_ir_v8(&self) -> &VerifiedCanonicalKernelIrV8 {
        match &self.canonical_kernel_ir {
            ProductionCanonicalKernelIrV1::V8(owner) => owner,
            ProductionCanonicalKernelIrV1::V9(_) => {
                panic!("Kernel IR V9 has no canonical V8 owner")
            }
        }
    }

    /// Borrows the typed identity of the authoritative canonical Kernel IR V8 bytes.
    pub const fn canonical_kernel_ir_v8_identity(&self) -> &VerifiedCanonicalKernelIrIdentityV8 {
        self.canonical_kernel_ir_v8().identity()
    }

    /// Borrows canonical V9 ownership when the lowered module uses the exact
    /// gfx950 collective or LDS transpose surface.
    pub const fn canonical_kernel_ir_v9(&self) -> Option<&VerifiedCanonicalKernelIrV9> {
        match &self.canonical_kernel_ir {
            ProductionCanonicalKernelIrV1::V8(_) => None,
            ProductionCanonicalKernelIrV1::V9(owner) => Some(owner),
        }
    }

    /// Borrows the typed V9 identity for an exact gfx950 collective or LDS transpose module.
    pub const fn canonical_kernel_ir_v9_identity(
        &self,
    ) -> Option<&VerifiedCanonicalKernelIrIdentityV9> {
        match self.canonical_kernel_ir_v9() {
            Some(owner) => Some(owner.identity()),
            None => None,
        }
    }

    /// Returns the exact version-bound identity of the retained canonical KIR.
    pub fn canonical_kernel_ir_identity(&self) -> ProductionCanonicalKernelIrIdentityV1 {
        self.canonical_kernel_ir.identity()
    }

    /// Borrows pointer-independent source correspondence evidence.
    pub const fn correspondence(&self) -> &SemanticKirCorrespondenceV1 {
        &self.correspondence
    }

    /// Reports whether mandatory ranked checks remain owned by this lowering.
    pub const fn retains_mandatory_generic_checks(&self) -> bool {
        self.generic_checks.is_some()
    }

    /// Borrows independently checked MIR-to-PLIRON translation evidence when
    /// this owner was constructed through the production ranked pipeline.
    pub const fn mir_pliron_translation_validation(
        &self,
    ) -> Option<&ProductionMirPlironTranslationValidationV1> {
        match &self.generic_checks {
            Some(checks) => Some(&checks.translation_validation),
            None => None,
        }
    }

    pub(crate) fn retained_generic_checks_discharge_unsupported_indices(
        &self,
        reasons: &[FormalMemoryIncompleteReason],
    ) -> Result<(), ProductionMemoryDischargeFailureV1> {
        let Some(checks) = &self.generic_checks else {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "verified Kernel IR does not retain mandatory ranked checks",
            ));
        };
        if !mandatory_generic_checks_are_clean(&checks.lowering) {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "a retained mandatory ranked check is not clean",
            ));
        }
        unsupported_indices_match_ranked_sources_result(
            &self.module,
            &self.correspondence,
            &checks.lowering,
            &checks.access_sources,
            reasons,
            self.limits.max_operations,
        )
    }

    pub(crate) fn retained_generic_checks_discharge_guarded_accesses(
        &self,
        guarded_locations: &[FunctionOperationLocation],
    ) -> Result<(), ProductionMemoryDischargeFailureV1> {
        let Some(checks) = &self.generic_checks else {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "verified Kernel IR does not retain mandatory ranked checks",
            ));
        };
        if !mandatory_generic_checks_are_clean(&checks.lowering) {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "a retained mandatory ranked check is not clean",
            ));
        }
        guarded_accesses_have_structural_bounds_result(
            &self.module,
            guarded_locations,
            self.limits.max_operations,
        )
    }

    pub(crate) fn retained_collective_lowering_discharges_workgroup_memory(
        &self,
        reasons: &[FormalMemoryIncompleteReason],
    ) -> Result<(), ProductionMemoryDischargeFailureV1> {
        if reasons.is_empty() || reasons.len() > self.limits.max_operations {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "workgroup-memory discharge received an empty or oversized reason set",
            ));
        }
        let [kernel] = self.module.kernels.as_slice() else {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "workgroup-memory discharge requires exactly one kernel",
            ));
        };
        let body = self
            .module
            .function(&kernel.entry)
            .and_then(|function| function.body.as_ref())
            .ok_or_else(|| {
                ProductionMemoryDischargeFailureV1::stage(
                    "workgroup-memory discharge cannot find the selected kernel body",
                )
            })?;
        let mut saw_allocation = false;
        let mut saw_pointer_transport = false;
        for reason in reasons {
            match reason {
                FormalMemoryIncompleteReason::UnsupportedMemoryEffect { location } => {
                    let operation = operation_at_location_v1(body, *location).ok_or_else(|| {
                        ProductionMemoryDischargeFailureV1::access(
                            *location,
                            "workgroup-memory effect location is absent from exact Kernel IR",
                        )
                    })?;
                    match &operation.kind {
                        OperationKind::WorkgroupMemory(_) => {
                            if !matches!(
                                self.terminator_intrinsic_at_location_v1(*location),
                                Some(
                                    SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent { .. }
                                        | SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate { .. }
                                )
                            ) {
                                return Err(ProductionMemoryDischargeFailureV1::access(
                                    *location,
                                    "workgroup allocation is not owned by exact dynamic-LDS lowering",
                                ));
                            }
                            saw_allocation = true;
                        }
                        OperationKind::WorkgroupBarrier(_) => {
                            if !matches!(
                                self.terminator_intrinsic_at_location_v1(*location),
                                Some(
                                    SemanticCompilerIntrinsicOperationV1::WorkgroupReduceSum { .. }
                                        | SemanticCompilerIntrinsicOperationV1::WorkgroupBarrier
                                        | SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineEvent {
                                            event: SemanticWorkgroupPipelineEventV1::Wait,
                                            ..
                                        }
                                )
                            ) {
                                return Err(ProductionMemoryDischargeFailureV1::access(
                                    *location,
                                    "workgroup barrier is not owned by exact reduction lowering",
                                ));
                            }
                        }
                        _ => {
                            return Err(ProductionMemoryDischargeFailureV1::access(
                                *location,
                                "unsupported memory effect is not an admitted compiler-owned workgroup operation",
                            ));
                        }
                    }
                }
                FormalMemoryIncompleteReason::UnsupportedPointerDerivation {
                    location,
                    pointer,
                } => {
                    let operation = operation_at_location_v1(body, *location).ok_or_else(|| {
                        ProductionMemoryDischargeFailureV1::access(
                            *location,
                            "workgroup pointer transport location is absent from exact Kernel IR",
                        )
                    })?;
                    let [result] = operation.results.as_slice() else {
                        return Err(ProductionMemoryDischargeFailureV1::access(
                            *location,
                            "workgroup pointer transport does not define exactly one value",
                        ));
                    };
                    let direct_allocation = matches!(
                        &operation.kind,
                        OperationKind::WorkgroupMemory(_)
                    ) && matches!(
                        self.terminator_intrinsic_at_location_v1(*location),
                        Some(
                            SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent { .. }
                                | SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate { .. }
                        )
                    );
                    let enum_transport = matches!(
                        &operation.kind,
                        OperationKind::Load { access, .. }
                            if access.address_space == AddressSpace::Private
                    ) && self.synthetic_rule_owns_location_v1(
                        *location,
                        SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage,
                    );
                    if result.id != *pointer
                        || !matches!(
                            &result.ty,
                            Type::Pointer(pointer)
                                if pointer.address_space == AddressSpace::Workgroup
                        )
                        || (!direct_allocation && !enum_transport)
                    {
                        return Err(ProductionMemoryDischargeFailureV1::access(
                            *location,
                            "workgroup pointer is not an exact compiler-owned enum transport",
                        ));
                    }
                    saw_allocation |= direct_allocation;
                    saw_pointer_transport |= enum_transport;
                }
                _ => {
                    return Err(ProductionMemoryDischargeFailureV1::stage(
                        "workgroup-memory discharge received another incomplete-reason kind",
                    ));
                }
            }
        }
        if saw_pointer_transport && !saw_allocation {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "workgroup pointer transport has no compiler-owned allocation effect",
            ));
        }
        Ok(())
    }

    fn terminator_intrinsic_at_location_v1(
        &self,
        location: FunctionOperationLocation,
    ) -> Option<SemanticCompilerIntrinsicOperationV1> {
        let span = self
            .correspondence
            .terminator_operation_spans()
            .iter()
            .find(|span| {
                operation_span_contains_v1(
                    span.kernel_ir_block(),
                    span.first_operation_ordinal(),
                    span.operation_count(),
                    location,
                )
            })?;
        let semantic = self.semantic.semantic();
        let function = semantic
            .functions()
            .get(span.semantic_function().index() as usize)?;
        let block = function
            .blocks()
            .get(span.semantic_block().index() as usize)?;
        let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
            return None;
        };
        let SemanticCallableDeclV1::CompilerIntrinsic { operation, .. } =
            semantic.callables().get(call.callee().index() as usize)?
        else {
            return None;
        };
        Some(*operation)
    }

    fn synthetic_rule_owns_location_v1(
        &self,
        location: FunctionOperationLocation,
        rule: SemanticKirSyntheticOperationRuleV1,
    ) -> bool {
        self.correspondence
            .synthetic_operation_spans()
            .iter()
            .any(|span| {
                span.rule() == rule
                    && operation_span_contains_v1(
                        span.kernel_ir_block(),
                        span.first_operation_ordinal(),
                        span.operation_count(),
                        location,
                    )
            })
    }

    /// Exact target-neutral lowering evidence is not artifact or launch authority.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

fn operation_at_location_v1(
    body: &FunctionBody,
    location: FunctionOperationLocation,
) -> Option<&Operation> {
    body.blocks
        .iter()
        .find(|block| block.id == location.block)
        .and_then(|block| block.operations.get(location.operation_index))
}

fn operation_span_contains_v1(
    block: BlockId,
    first: u32,
    count: u32,
    location: FunctionOperationLocation,
) -> bool {
    if location.block != block {
        return false;
    }
    let Ok(operation) = u32::try_from(location.operation_index) else {
        return false;
    };
    first
        .checked_add(count)
        .is_some_and(|end| operation >= first && operation < end)
}

fn ranked_access_sources_are_well_formed(
    lowering: &ProductionRankedKernelLoweringInputV1,
    sources: &[ProductionRankedAccessSourceV1],
) -> bool {
    if sources.len() > DEFAULT_MAX_OPERATIONS_V1 {
        return false;
    }
    let mut ranked_locations = BTreeSet::new();
    let mut source_ordinals = BTreeMap::<(u32, Option<u32>), BTreeSet<u32>>::new();
    for source in sources {
        let Some(operation) = lowering
            .kernel()
            .blocks()
            .get(source.ranked_block as usize)
            .and_then(|block| block.operations().get(source.ranked_operation as usize))
        else {
            return false;
        };
        if !matches!(
            operation,
            ProductionRankedOperationV1::Access { .. }
                | ProductionRankedOperationV1::PredicatedAccess { .. }
                | ProductionRankedOperationV1::ValueAccess { .. }
                | ProductionRankedOperationV1::AtomicAccess { .. }
                | ProductionRankedOperationV1::AtomicValueAccess { .. }
                | ProductionRankedOperationV1::AllocationEffect { .. }
        ) || !ranked_locations.insert((source.ranked_block, source.ranked_operation))
        {
            return false;
        }
        if !source_ordinals
            .entry((source.semantic_block, source.semantic_statement))
            .or_default()
            .insert(source.semantic_access_ordinal)
        {
            return false;
        }
    }
    source_ordinals.values().all(|ordinals| {
        ordinals
            .iter()
            .copied()
            .eq(0..u32::try_from(ordinals.len()).unwrap_or(u32::MAX))
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticAccessSiteV1 {
    block: u32,
    statement: Option<u32>,
    ordinal: u32,
}

// All correlation indexing and dependency propagation share this aggregate cap.
const UNSUPPORTED_INDEX_CORRELATION_STEPS_PER_OPERATION_V1: usize = 64;

struct UnsupportedIndexCorrelationBudgetV1 {
    remaining: usize,
}

impl UnsupportedIndexCorrelationBudgetV1 {
    fn charge(&mut self) -> Option<()> {
        self.remaining = self.remaining.checked_sub(1)?;
        Some(())
    }
}

#[derive(Clone, Copy)]
struct KirMemoryConsumerV1 {
    location: FunctionOperationLocation,
    operation_access_ordinal: u32,
    pointer: ValueId,
    access: dialect_kernel::AccessKindAttr,
    memory_space: dialect_kernel::MemorySpaceAttr,
    atomic: Option<NormalizedAtomicContractV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NormalizedAtomicContractV1 {
    ordering: u8,
    scope: u8,
    failure_ordering: Option<u8>,
}

struct KirCorrelationIndexV1<'module> {
    blocks: BTreeMap<BlockId, &'module [Operation]>,
    operations: BTreeMap<FunctionOperationLocation, &'module Operation>,
    definitions: BTreeMap<ValueId, &'module Operation>,
    definition_locations: BTreeMap<ValueId, FunctionOperationLocation>,
    pointer_dependents: BTreeMap<ValueId, BTreeSet<ValueId>>,
    memory_consumers: Vec<KirMemoryConsumerV1>,
    unmodeled_memory_effects: Vec<FunctionOperationLocation>,
}

#[derive(Clone, Copy)]
struct RankedViewDefinitionV1 {
    allocation_origin: u64,
    memory_space: dialect_kernel::MemorySpaceAttr,
    noalias_class: u64,
}

#[derive(Clone, Copy)]
enum IndexedRankedAllocationV1 {
    View(ProductionRankedValueV1),
    Direct(RankedViewDefinitionV1),
}

#[derive(Clone, Copy)]
struct IndexedRankedAccessSourceV1 {
    ranked_block: u32,
    ranked_operation: u32,
    access: dialect_kernel::AccessKindAttr,
    allocation: IndexedRankedAllocationV1,
    value: Option<ProductionRankedValueV1>,
    atomic: Option<NormalizedAtomicContractV1>,
}

struct RankedCorrelationIndexV1 {
    sources_by_site: BTreeMap<SemanticAccessSiteV1, IndexedRankedAccessSourceV1>,
    conservative_sources_by_statement: BTreeMap<(u32, Option<u32>), IndexedRankedAccessSourceV1>,
    sites_by_ranked_location: BTreeMap<(u32, u32), SemanticAccessSiteV1>,
    view_definitions: BTreeMap<ProductionRankedValueIdV1, RankedViewDefinitionV1>,
    semantic_expressions: BTreeMap<
        ProductionRankedValueIdV1,
        (
            ProductionSemanticExpressionV2,
            ProductionNumericalContractV2,
        ),
    >,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Exact failure at the ranked-check/Kernel-IR memory-proof boundary.
pub enum ProductionMemoryDischargeFailureV1 {
    /// A module-wide correlation or resource invariant failed.
    Stage(&'static str),
    /// One exact Kernel IR access failed correlation.
    Access {
        /// Location of the failing access or pointer operation.
        location: FunctionOperationLocation,
        /// Stable failure classification.
        detail: &'static str,
    },
    /// A guarded access lacks proof of its exact selected index bound.
    GuardedBound {
        /// Location of the failing guarded load.
        location: FunctionOperationLocation,
        /// Index selected when the guard is true.
        index: ValueId,
        /// Slice whose dynamic length must bound `index`.
        slice: ValueId,
        /// Stable failure classification.
        detail: &'static str,
    },
}

impl ProductionMemoryDischargeFailureV1 {
    const fn stage(detail: &'static str) -> Self {
        Self::Stage(detail)
    }

    const fn access(location: FunctionOperationLocation, detail: &'static str) -> Self {
        Self::Access { location, detail }
    }

    const fn guarded_bound(
        location: FunctionOperationLocation,
        index: ValueId,
        slice: ValueId,
        detail: &'static str,
    ) -> Self {
        Self::GuardedBound {
            location,
            index,
            slice,
            detail,
        }
    }
}

impl fmt::Display for ProductionMemoryDischargeFailureV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stage(detail) => formatter.write_str(detail),
            Self::Access { location, detail } => {
                write!(formatter, "{detail} at {location:?}")
            }
            Self::GuardedBound {
                location,
                index,
                slice,
                detail,
            } => write!(
                formatter,
                "{detail} at {location:?}: index {index:?} must be below the length of slice {slice:?}",
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct IndexedUnsupportedReasonV1 {
    pointer: ValueId,
    allocation_parameter: u32,
    location: FunctionOperationLocation,
}

fn unsupported_indices_match_ranked_sources_result(
    module: &Module,
    correspondence: &SemanticKirCorrespondenceV1,
    lowering: &ProductionRankedKernelLoweringInputV1,
    sources: &[ProductionRankedAccessSourceV1],
    reasons: &[FormalMemoryIncompleteReason],
    max_operations: usize,
) -> Result<(), ProductionMemoryDischargeFailureV1> {
    if reasons.len() > max_operations || sources.len() > max_operations {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation exceeded its input resource limit",
        ));
    }
    let [kernel] = module.kernels.as_slice() else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation requires exactly one kernel",
        ));
    };
    let Some(function) = module
        .functions
        .iter()
        .find(|function| function.id == kernel.entry)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation cannot find the selected kernel entry",
        ));
    };
    let Some(body) = function.body.as_ref() else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation selected a kernel declaration without a body",
        ));
    };
    let Some(steps) =
        max_operations.checked_mul(UNSUPPORTED_INDEX_CORRELATION_STEPS_PER_OPERATION_V1)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation resource budget overflowed",
        ));
    };
    let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: steps };
    let Some(kir) = build_kir_correlation_index(body, max_operations, &mut budget) else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation could not index exact Kernel IR",
        ));
    };
    if let Some(location) = kir.unmodeled_memory_effects.first().copied() {
        return Err(ProductionMemoryDischargeFailureV1::access(
            location,
            "Kernel IR memory effect has no exact ranked access model",
        ));
    }
    let Some(semantic_sites) = index_semantic_access_sites(correspondence, &kir, &mut budget)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation could not index semantic access sites",
        ));
    };
    let Some(ranked) = index_ranked_correlation(lowering, sources, max_operations, &mut budget)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation could not index ranked access receipts",
        ));
    };

    let mut indexed_reasons = Vec::with_capacity(reasons.len());
    let mut reason_pointers = BTreeSet::new();
    for reason in reasons {
        let FormalMemoryIncompleteReason::UnsupportedIndexExpression {
            location,
            index,
            allocation,
        } = reason
        else {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "unsupported-index correlation received another incomplete-reason kind",
            ));
        };
        let Some(defining_operation) = kir.operations.get(location) else {
            return Err(ProductionMemoryDischargeFailureV1::access(
                *location,
                "unsupported index location is absent from exact Kernel IR",
            ));
        };
        let OperationKind::GetElementPointer { offset, .. } = defining_operation.kind else {
            return Err(ProductionMemoryDischargeFailureV1::access(
                *location,
                "unsupported index location is not a pointer-offset operation",
            ));
        };
        let [pointer] = defining_operation.results.as_slice() else {
            return Err(ProductionMemoryDischargeFailureV1::access(
                *location,
                "unsupported pointer offset does not define exactly one pointer",
            ));
        };
        if offset != *index
            || !kir
                .definitions
                .get(&pointer.id)
                .is_some_and(|definition| std::ptr::eq(*definition, *defining_operation))
            || budget.charge().is_none()
        {
            return Err(ProductionMemoryDischargeFailureV1::access(
                *location,
                "unsupported index does not match its exact pointer definition",
            ));
        }
        indexed_reasons.push(IndexedUnsupportedReasonV1 {
            pointer: pointer.id,
            allocation_parameter: allocation.parameter_index(),
            location: *location,
        });
        reason_pointers.insert(pointer.id);
    }
    let Some(consumers_by_pointer) =
        propagate_pointer_consumers(&kir, &reason_pointers, &mut budget)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "unsupported-index correlation could not propagate pointer consumers",
        ));
    };

    let mut used_ranked_locations = BTreeSet::new();
    for reason in indexed_reasons {
        if budget.charge().is_none() {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "unsupported-index correlation exhausted its work budget",
            ));
        }
        let Some(consumers) = consumers_by_pointer.get(&reason.pointer) else {
            return Err(ProductionMemoryDischargeFailureV1::access(
                reason.location,
                "unsupported pointer has no memory consumer",
            ));
        };
        if consumers.is_empty() {
            return Err(ProductionMemoryDischargeFailureV1::access(
                reason.location,
                "unsupported pointer has no memory consumer",
            ));
        }
        for consumer in consumers {
            let Some(site) =
                semantic_sites.get(&(consumer.location, consumer.operation_access_ordinal))
            else {
                return Err(ProductionMemoryDischargeFailureV1::access(
                    consumer.location,
                    "Kernel IR memory consumer has no exact semantic access site",
                ));
            };
            let Some((_logical_site, source)) =
                ranked_source_for_semantic_effect_v1(&ranked, *site)
            else {
                return Err(ProductionMemoryDischargeFailureV1::access(
                    consumer.location,
                    "semantic memory access site has no ranked access receipt",
                ));
            };
            let first_logical_use =
                used_ranked_locations.insert((source.ranked_block, source.ranked_operation));
            if (!first_logical_use
                && matches!(source.allocation, IndexedRankedAllocationV1::View(_)))
                || !indexed_ranked_source_matches_allocation(
                    &ranked,
                    source,
                    consumer.access,
                    consumer.memory_space,
                    reason.allocation_parameter,
                )
            {
                return Err(ProductionMemoryDischargeFailureV1::access(
                    consumer.location,
                    "ranked access receipt does not match the exact allocation or access kind",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn unsupported_indices_match_ranked_sources(
    module: &Module,
    correspondence: &SemanticKirCorrespondenceV1,
    lowering: &ProductionRankedKernelLoweringInputV1,
    sources: &[ProductionRankedAccessSourceV1],
    reasons: &[FormalMemoryIncompleteReason],
    max_operations: usize,
) -> bool {
    unsupported_indices_match_ranked_sources_result(
        module,
        correspondence,
        lowering,
        sources,
        reasons,
        max_operations,
    )
    .is_ok()
}

fn build_kir_correlation_index<'module>(
    body: &'module FunctionBody,
    max_operations: usize,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<KirCorrelationIndexV1<'module>> {
    let mut blocks = BTreeMap::new();
    let mut operations = BTreeMap::new();
    let mut definitions = BTreeMap::new();
    let mut definition_locations = BTreeMap::new();
    let mut memory_consumers = Vec::new();
    let mut unmodeled_memory_effects = Vec::new();
    let mut operation_count = 0_usize;
    for block in &body.blocks {
        budget.charge()?;
        if blocks
            .insert(block.id, block.operations.as_slice())
            .is_some()
        {
            return None;
        }
        for (operation_index, operation) in block.operations.iter().enumerate() {
            operation_count = operation_count.checked_add(1)?;
            if operation_count > max_operations {
                return None;
            }
            budget.charge()?;
            let location = FunctionOperationLocation::new(block.id, operation_index);
            if operations.insert(location, operation).is_some() {
                return None;
            }
            for result in &operation.results {
                budget.charge()?;
                if definitions.insert(result.id, operation).is_some()
                    || definition_locations.insert(result.id, location).is_some()
                {
                    return None;
                }
            }
            let accesses = kir_memory_accesses_v1(operation);
            let executable_effects = operation
                .memory_effects()
                .into_iter()
                .filter(|effect| {
                    matches!(
                        effect,
                        MemoryEffect::Read(_)
                            | MemoryEffect::Write(_)
                            | MemoryEffect::VolatileRead(_)
                            | MemoryEffect::VolatileWrite(_)
                            | MemoryEffect::Atomic { .. }
                    )
                })
                .count();
            let has_unmodeled_inline_effect = matches!(
                &operation.kind,
                OperationKind::InlineAssembly(assembly) if !assembly.declared_effects.is_empty()
            );
            if accesses.len() != executable_effects || has_unmodeled_inline_effect {
                budget.charge()?;
                unmodeled_memory_effects.push(location);
            }
            for (operation_access_ordinal, (pointer, access, memory_space, atomic)) in
                accesses.into_iter().enumerate()
            {
                budget.charge()?;
                memory_consumers.push(KirMemoryConsumerV1 {
                    location,
                    operation_access_ordinal: u32::try_from(operation_access_ordinal).ok()?,
                    pointer,
                    access,
                    memory_space,
                    atomic,
                });
            }
        }
    }

    let mut pointer_dependents = BTreeMap::<ValueId, BTreeSet<ValueId>>::new();
    let mut pointer_indegree = BTreeMap::<ValueId, usize>::new();
    for operation in operations.values() {
        budget.charge()?;
        let dependencies = match &operation.kind {
            OperationKind::GetElementPointer { base, .. } => Some([Some(*base), None]),
            OperationKind::Cast { value, .. } => Some([Some(*value), None]),
            OperationKind::Select {
                true_value,
                false_value,
                ..
            } => Some([Some(*true_value), Some(*false_value)]),
            _ => None,
        };
        let Some(dependencies) = dependencies else {
            continue;
        };
        for result in &operation.results {
            for dependency in dependencies.into_iter().flatten() {
                budget.charge()?;
                if pointer_dependents
                    .entry(dependency)
                    .or_default()
                    .insert(result.id)
                {
                    budget.charge()?;
                    pointer_indegree.entry(dependency).or_insert(0);
                    let degree = pointer_indegree.entry(result.id).or_insert(0);
                    *degree = degree.checked_add(1)?;
                }
            }
        }
    }
    if pointer_dependency_graph_has_cycle(&pointer_dependents, pointer_indegree, budget)? {
        return None;
    }
    Some(KirCorrelationIndexV1 {
        blocks,
        operations,
        definitions,
        definition_locations,
        pointer_dependents,
        memory_consumers,
        unmodeled_memory_effects,
    })
}

fn pointer_dependency_graph_has_cycle(
    outgoing: &BTreeMap<ValueId, BTreeSet<ValueId>>,
    mut indegree: BTreeMap<ValueId, usize>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<bool> {
    let mut ready = BTreeSet::new();
    for (value, degree) in &indegree {
        budget.charge()?;
        if *degree == 0 {
            ready.insert(*value);
        }
    }
    let mut visited = 0_usize;
    while let Some(value) = ready.pop_first() {
        budget.charge()?;
        visited = visited.checked_add(1)?;
        for result in outgoing.get(&value).into_iter().flatten() {
            budget.charge()?;
            let degree = indegree.get_mut(result)?;
            *degree = degree.checked_sub(1)?;
            if *degree == 0 {
                ready.insert(*result);
            }
        }
    }
    Some(visited != indegree.len())
}

fn propagate_pointer_consumers(
    kir: &KirCorrelationIndexV1<'_>,
    roots: &BTreeSet<ValueId>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<BTreeMap<ValueId, Vec<KirMemoryConsumerV1>>> {
    let mut roots_by_value = BTreeMap::<ValueId, BTreeSet<ValueId>>::new();
    let mut worklist = VecDeque::new();
    for root in roots {
        budget.charge()?;
        roots_by_value.entry(*root).or_default().insert(*root);
        worklist.push_back((*root, *root));
    }
    while let Some((value, root)) = worklist.pop_front() {
        budget.charge()?;
        for dependent in kir.pointer_dependents.get(&value).into_iter().flatten() {
            budget.charge()?;
            if roots_by_value.entry(*dependent).or_default().insert(root) {
                budget.charge()?;
                worklist.push_back((*dependent, root));
            }
        }
    }
    let mut consumers_by_root = BTreeMap::<ValueId, Vec<KirMemoryConsumerV1>>::new();
    for consumer in &kir.memory_consumers {
        budget.charge()?;
        let Some(dependencies) = roots_by_value.get(&consumer.pointer) else {
            continue;
        };
        for root in dependencies {
            budget.charge()?;
            consumers_by_root.entry(*root).or_default().push(*consumer);
        }
    }
    Some(consumers_by_root)
}

fn kir_memory_accesses_v1(
    operation: &Operation,
) -> Vec<(
    ValueId,
    dialect_kernel::AccessKindAttr,
    dialect_kernel::MemorySpaceAttr,
    Option<NormalizedAtomicContractV1>,
)> {
    let one = |pointer, access, address_space, atomic| {
        ranked_memory_space(address_space)
            .map(|space| vec![(pointer, access, space, atomic)])
            .unwrap_or_default()
    };
    match &operation.kind {
        OperationKind::Load { pointer, access }
        | OperationKind::GuardedLoad {
            pointer, access, ..
        } => one(
            *pointer,
            dialect_kernel::AccessKindAttr::Read,
            access.address_space,
            None,
        ),
        OperationKind::Store {
            pointer, access, ..
        } => one(
            *pointer,
            dialect_kernel::AccessKindAttr::Write,
            access.address_space,
            None,
        ),
        OperationKind::Atomic(atomic) => {
            let kind = match atomic.kind {
                AtomicKind::Load => dialect_kernel::AccessKindAttr::AtomicRead,
                AtomicKind::Store => dialect_kernel::AccessKindAttr::AtomicWrite,
                AtomicKind::Exchange
                | AtomicKind::CompareExchange
                | AtomicKind::Add
                | AtomicKind::Subtract
                | AtomicKind::Min
                | AtomicKind::Max
                | AtomicKind::BitAnd
                | AtomicKind::BitOr
                | AtomicKind::BitXor => dialect_kernel::AccessKindAttr::AtomicReadModifyWrite,
            };
            let Some(scope) = normalize_kir_atomic_scope_v1(atomic.scope) else {
                return Vec::new();
            };
            one(
                atomic.pointer,
                kind,
                atomic.access.address_space,
                Some(NormalizedAtomicContractV1 {
                    ordering: normalize_kir_atomic_ordering_v1(atomic.ordering),
                    scope,
                    failure_ordering: atomic
                        .failure_ordering
                        .map(normalize_kir_atomic_ordering_v1),
                }),
            )
        }
        OperationKind::MemoryIntrinsic(intrinsic) => match intrinsic {
            MemoryIntrinsicOperation::PointerDistance { .. }
            | MemoryIntrinsicOperation::VolatileLoad { .. }
            | MemoryIntrinsicOperation::VolatileStore { .. } => Vec::new(),
            MemoryIntrinsicOperation::CopyNonOverlapping {
                source,
                destination,
                source_address_space,
                destination_address_space,
                ..
            } => {
                let mut effects = one(
                    *source,
                    dialect_kernel::AccessKindAttr::Read,
                    *source_address_space,
                    None,
                );
                effects.extend(one(
                    *destination,
                    dialect_kernel::AccessKindAttr::Write,
                    *destination_address_space,
                    None,
                ));
                effects
            }
        },
        OperationKind::Matrix(matrix) => match matrix.kind {
            MatrixOperationKind::LdsLoad { base, .. } => one(
                base,
                dialect_kernel::AccessKindAttr::Read,
                AddressSpace::Workgroup,
                None,
            ),
            MatrixOperationKind::LdsStore { base, .. } => one(
                base,
                dialect_kernel::AccessKindAttr::Write,
                AddressSpace::Workgroup,
                None,
            ),
            MatrixOperationKind::MultiplyAccumulate { .. }
            | MatrixOperationKind::ScaledMultiplyAccumulate { .. } => Vec::new(),
        },
        OperationKind::Gfx950LdsTranspose(transpose) => match transpose.kind {
            Gfx950LdsTransposeOperationKindV1::Stage {
                storage,
                source_slice,
                ..
            } => vec![
                (
                    source_slice,
                    dialect_kernel::AccessKindAttr::Read,
                    dialect_kernel::MemorySpaceAttr::Global,
                    None,
                ),
                (
                    storage,
                    dialect_kernel::AccessKindAttr::Write,
                    dialect_kernel::MemorySpaceAttr::Workgroup,
                    None,
                ),
            ],
            Gfx950LdsTransposeOperationKindV1::Read { storage, .. } => one(
                storage,
                dialect_kernel::AccessKindAttr::Read,
                AddressSpace::Workgroup,
                None,
            ),
            Gfx950LdsTransposeOperationKindV1::Current { .. }
            | Gfx950LdsTransposeOperationKindV1::Publish { .. } => Vec::new(),
        },
        _ => Vec::new(),
    }
}

const fn ranked_memory_space(
    address_space: AddressSpace,
) -> Option<dialect_kernel::MemorySpaceAttr> {
    match address_space {
        AddressSpace::Private => Some(dialect_kernel::MemorySpaceAttr::Private),
        AddressSpace::Workgroup => Some(dialect_kernel::MemorySpaceAttr::Workgroup),
        AddressSpace::Global => Some(dialect_kernel::MemorySpaceAttr::Global),
        AddressSpace::Constant | AddressSpace::Generic => None,
    }
}

const fn normalize_kir_atomic_ordering_v1(ordering: MemoryOrdering) -> u8 {
    match ordering {
        MemoryOrdering::Relaxed => 0,
        MemoryOrdering::Acquire => 1,
        MemoryOrdering::Release => 2,
        MemoryOrdering::AcquireRelease => 3,
        MemoryOrdering::SequentiallyConsistent => 4,
    }
}

const fn normalize_kir_atomic_scope_v1(scope: SynchronizationScope) -> Option<u8> {
    match scope {
        SynchronizationScope::Invocation => Some(0),
        SynchronizationScope::Workgroup => Some(1),
        // The semantic MIR Agent scope is the supported source of Kernel IR
        // Device scope in this target-neutral lowering.
        SynchronizationScope::Device => Some(2),
        SynchronizationScope::System => Some(4),
        SynchronizationScope::Subgroup => None,
    }
}

const fn normalize_ranked_atomic_contract_v1(
    ordering: dialect_kernel::AtomicOrderingAttr,
    scope: dialect_kernel::AtomicScopeAttr,
) -> NormalizedAtomicContractV1 {
    let ordering = match ordering {
        dialect_kernel::AtomicOrderingAttr::Relaxed => 0,
        dialect_kernel::AtomicOrderingAttr::Acquire => 1,
        dialect_kernel::AtomicOrderingAttr::Release => 2,
        dialect_kernel::AtomicOrderingAttr::AcquireRelease => 3,
        dialect_kernel::AtomicOrderingAttr::SequentiallyConsistent => 4,
    };
    NormalizedAtomicContractV1 {
        ordering,
        scope: scope.rank(),
        failure_ordering: None,
    }
}

fn index_semantic_access_sites(
    correspondence: &SemanticKirCorrespondenceV1,
    kir: &KirCorrelationIndexV1<'_>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1>> {
    let mut sites = BTreeMap::new();
    for span in correspondence.statement_operation_spans() {
        budget.charge()?;
        index_semantic_access_span(
            kir,
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            span.semantic_block().index(),
            Some(span.statement_ordinal()),
            &mut sites,
            budget,
        )?;
    }
    for span in correspondence.terminator_operation_spans() {
        budget.charge()?;
        index_semantic_access_span(
            kir,
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            span.semantic_block().index(),
            None,
            &mut sites,
            budget,
        )?;
    }
    Some(sites)
}

#[allow(clippy::too_many_arguments)]
fn index_semantic_access_span(
    kir: &KirCorrelationIndexV1<'_>,
    block: BlockId,
    first: u32,
    count: u32,
    semantic_block: u32,
    semantic_statement: Option<u32>,
    sites: &mut BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<()> {
    let operations = kir.blocks.get(&block)?;
    let first = first as usize;
    let end = first.checked_add(count as usize)?;
    let operations = operations.get(first..end)?;
    let mut access_ordinal = 0_u32;
    for (relative_ordinal, operation) in operations.iter().enumerate() {
        budget.charge()?;
        let operation_index = first.checked_add(relative_ordinal)?;
        let location = FunctionOperationLocation::new(block, operation_index);
        for operation_access_ordinal in 0..kir_memory_accesses_v1(operation).len() {
            budget.charge()?;
            let site = SemanticAccessSiteV1 {
                block: semantic_block,
                statement: semantic_statement,
                ordinal: access_ordinal,
            };
            let operation_access_ordinal = u32::try_from(operation_access_ordinal).ok()?;
            if sites
                .insert((location, operation_access_ordinal), site)
                .is_some()
            {
                return None;
            }
            access_ordinal = access_ordinal.checked_add(1)?;
        }
    }
    Some(())
}

fn index_ranked_correlation(
    lowering: &ProductionRankedKernelLoweringInputV1,
    sources: &[ProductionRankedAccessSourceV1],
    max_operations: usize,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<RankedCorrelationIndexV1> {
    if sources.len() > DEFAULT_MAX_OPERATIONS_V1 {
        return None;
    }
    let mut operation_count = 0_usize;
    let mut view_definitions = BTreeMap::new();
    let mut semantic_expressions = BTreeMap::new();
    for block in lowering.kernel().blocks() {
        budget.charge()?;
        for operation in block.operations() {
            operation_count = operation_count.checked_add(1)?;
            if operation_count > max_operations {
                return None;
            }
            budget.charge()?;
            let definition = match operation {
                ProductionRankedOperationV1::View {
                    result,
                    allocation_origin,
                    ..
                } => Some((
                    *result,
                    RankedViewDefinitionV1 {
                        allocation_origin: *allocation_origin,
                        memory_space: dialect_kernel::MemorySpaceAttr::Global,
                        noalias_class: 0,
                    },
                )),
                ProductionRankedOperationV1::ViewInSpace {
                    result,
                    memory_space,
                    allocation_origin,
                    ..
                } => Some((
                    *result,
                    RankedViewDefinitionV1 {
                        allocation_origin: *allocation_origin,
                        memory_space: *memory_space,
                        noalias_class: 0,
                    },
                )),
                _ => None,
            };
            if let Some((result, definition)) = definition {
                budget.charge()?;
                if view_definitions.insert(result, definition).is_some() {
                    return None;
                }
            }
            if let ProductionRankedOperationV1::SemanticExpression {
                result,
                expression,
                numerical_contract,
            } = operation
            {
                budget.charge()?;
                if semantic_expressions
                    .insert(*result, (expression.clone(), *numerical_contract))
                    .is_some()
                {
                    return None;
                }
            }
        }
    }

    let mut ranked_locations = BTreeSet::new();
    let mut sites_by_ranked_location = BTreeMap::new();
    let mut source_ordinals = BTreeMap::<(u32, Option<u32>), BTreeSet<u32>>::new();
    let mut sources_by_site = BTreeMap::new();
    let mut conservative_sources_by_statement = BTreeMap::new();
    let mut ambiguous_conservative_statements = BTreeSet::new();
    for source in sources {
        budget.charge()?;
        let operation = lowering
            .kernel()
            .blocks()
            .get(source.ranked_block as usize)?
            .operations()
            .get(source.ranked_operation as usize)?;
        let (access, allocation, value, atomic) = match operation {
            ProductionRankedOperationV1::Access { kind, view, .. } => {
                (*kind, IndexedRankedAllocationV1::View(*view), None, None)
            }
            ProductionRankedOperationV1::PredicatedAccess { kind, view, .. } => {
                (*kind, IndexedRankedAllocationV1::View(*view), None, None)
            }
            ProductionRankedOperationV1::ValueAccess {
                kind, view, value, ..
            } => (
                *kind,
                IndexedRankedAllocationV1::View(*view),
                Some(*value),
                None,
            ),
            ProductionRankedOperationV1::AtomicAccess {
                kind,
                ordering,
                scope,
                view,
                ..
            } => (
                *kind,
                IndexedRankedAllocationV1::View(*view),
                None,
                Some(normalize_ranked_atomic_contract_v1(*ordering, *scope)),
            ),
            ProductionRankedOperationV1::AtomicValueAccess {
                kind,
                ordering,
                scope,
                view,
                value,
                ..
            } => (
                *kind,
                IndexedRankedAllocationV1::View(*view),
                Some(*value),
                Some(normalize_ranked_atomic_contract_v1(*ordering, *scope)),
            ),
            ProductionRankedOperationV1::AllocationEffect {
                kind,
                memory_space,
                allocation_origin,
                noalias_class,
            } => (
                *kind,
                IndexedRankedAllocationV1::Direct(RankedViewDefinitionV1 {
                    allocation_origin: *allocation_origin,
                    memory_space: *memory_space,
                    noalias_class: *noalias_class,
                }),
                None,
                None,
            ),
            _ => return None,
        };
        if !ranked_locations.insert((source.ranked_block, source.ranked_operation))
            || !source_ordinals
                .entry((source.semantic_block, source.semantic_statement))
                .or_default()
                .insert(source.semantic_access_ordinal)
        {
            return None;
        }
        let site = SemanticAccessSiteV1 {
            block: source.semantic_block,
            statement: source.semantic_statement,
            ordinal: source.semantic_access_ordinal,
        };
        if sites_by_ranked_location
            .insert((source.ranked_block, source.ranked_operation), site)
            .is_some()
        {
            return None;
        }
        let indexed = IndexedRankedAccessSourceV1 {
            ranked_block: source.ranked_block,
            ranked_operation: source.ranked_operation,
            access,
            allocation,
            value,
            atomic,
        };
        if matches!(allocation, IndexedRankedAllocationV1::Direct(_)) {
            let key = (site.block, site.statement);
            if !ambiguous_conservative_statements.contains(&key)
                && conservative_sources_by_statement
                    .insert(key, indexed)
                    .is_some()
            {
                conservative_sources_by_statement.remove(&key);
                ambiguous_conservative_statements.insert(key);
            }
        }
        if sources_by_site.insert(site, indexed).is_some() {
            return None;
        }
    }
    for ordinals in source_ordinals.values() {
        budget.charge()?;
        if !ordinals
            .iter()
            .copied()
            .eq(0..u32::try_from(ordinals.len()).unwrap_or(u32::MAX))
        {
            return None;
        }
    }
    Some(RankedCorrelationIndexV1 {
        sources_by_site,
        conservative_sources_by_statement,
        sites_by_ranked_location,
        view_definitions,
        semantic_expressions,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NormalizedScalarExpressionV1 {
    Symbol {
        symbol: u32,
        scalar: ProductionSemanticScalarTypeV2,
    },
    Constant {
        scalar: ProductionSemanticScalarTypeV2,
        bits: u64,
    },
    Load {
        site: SemanticAccessSiteV1,
        scalar: ProductionSemanticScalarTypeV2,
    },
    Unary {
        operation: ProductionSemanticUnaryOpV2,
        scalar: ProductionSemanticScalarTypeV2,
        operand: Box<Self>,
    },
    Binary {
        operation: ProductionSemanticBinaryOpV2,
        scalar: ProductionSemanticScalarTypeV2,
        overflow: ProductionOverflowContractV2,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Compare {
        operation: ProductionSemanticComparisonV2,
        operand_scalar: ProductionSemanticScalarTypeV2,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Select {
        scalar: ProductionSemanticScalarTypeV2,
        condition: Box<Self>,
        when_true: Box<Self>,
        when_false: Box<Self>,
    },
    Cast {
        kind: ProductionSemanticCastV2,
        source: ProductionSemanticScalarTypeV2,
        target: ProductionSemanticScalarTypeV2,
        operand: Box<Self>,
    },
}

fn normalize_ranked_expression_v1(
    expression: &ProductionSemanticExpressionV2,
    lowering: &ProductionRankedKernelLoweringInputV1,
    ranked: &RankedCorrelationIndexV1,
    depth: usize,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<NormalizedScalarExpressionV1> {
    budget.charge()?;
    if depth > MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 {
        return None;
    }
    let next = depth.checked_add(1)?;
    Some(match expression {
        ProductionSemanticExpressionV2::Symbol { symbol, scalar } => {
            NormalizedScalarExpressionV1::Symbol {
                symbol: *symbol,
                scalar: *scalar,
            }
        }
        ProductionSemanticExpressionV2::Constant { scalar, bits } => {
            NormalizedScalarExpressionV1::Constant {
                scalar: *scalar,
                bits: *bits,
            }
        }
        ProductionSemanticExpressionV2::Load(load) => {
            let site = *ranked
                .sites_by_ranked_location
                .get(&(load.block, load.operation))?;
            let source = ranked.sources_by_site.get(&site)?;
            if source.access != dialect_kernel::AccessKindAttr::Read {
                return None;
            }
            let IndexedRankedAllocationV1::View(source_view) = source.allocation else {
                return None;
            };
            if source_view != load.view {
                return None;
            }
            let ProductionRankedValueV1::Local(view) = source_view else {
                return None;
            };
            let definition = ranked.view_definitions.get(&view)?;
            if definition.memory_space != dialect_kernel::MemorySpaceAttr::Global
                || definition.allocation_origin != load.allocation_origin
            {
                return None;
            }
            let operation = lowering
                .kernel()
                .blocks()
                .get(load.block as usize)?
                .operations()
                .get(load.operation as usize)?;
            let indices = match operation {
                ProductionRankedOperationV1::Access { indices, .. }
                | ProductionRankedOperationV1::ValueAccess { indices, .. }
                | ProductionRankedOperationV1::AtomicAccess { indices, .. }
                | ProductionRankedOperationV1::AtomicValueAccess { indices, .. } => indices,
                _ => return None,
            };
            if indices.as_slice() != load.indices.as_ref() {
                return None;
            }
            NormalizedScalarExpressionV1::Load {
                site,
                scalar: load.scalar,
            }
        }
        ProductionSemanticExpressionV2::Unary {
            operation,
            scalar,
            operand,
        } => NormalizedScalarExpressionV1::Unary {
            operation: *operation,
            scalar: *scalar,
            operand: Box::new(normalize_ranked_expression_v1(
                operand, lowering, ranked, next, budget,
            )?),
        },
        ProductionSemanticExpressionV2::Binary {
            operation,
            scalar,
            overflow,
            lhs,
            rhs,
        } => NormalizedScalarExpressionV1::Binary {
            operation: *operation,
            scalar: *scalar,
            overflow: *overflow,
            lhs: Box::new(normalize_ranked_expression_v1(
                lhs, lowering, ranked, next, budget,
            )?),
            rhs: Box::new(normalize_ranked_expression_v1(
                rhs, lowering, ranked, next, budget,
            )?),
        },
        ProductionSemanticExpressionV2::Compare {
            operation,
            operand_scalar,
            lhs,
            rhs,
        } => NormalizedScalarExpressionV1::Compare {
            operation: *operation,
            operand_scalar: *operand_scalar,
            lhs: Box::new(normalize_ranked_expression_v1(
                lhs, lowering, ranked, next, budget,
            )?),
            rhs: Box::new(normalize_ranked_expression_v1(
                rhs, lowering, ranked, next, budget,
            )?),
        },
        ProductionSemanticExpressionV2::Select {
            scalar,
            condition,
            when_true,
            when_false,
        } => NormalizedScalarExpressionV1::Select {
            scalar: *scalar,
            condition: Box::new(normalize_ranked_expression_v1(
                condition, lowering, ranked, next, budget,
            )?),
            when_true: Box::new(normalize_ranked_expression_v1(
                when_true, lowering, ranked, next, budget,
            )?),
            when_false: Box::new(normalize_ranked_expression_v1(
                when_false, lowering, ranked, next, budget,
            )?),
        },
        ProductionSemanticExpressionV2::Cast {
            kind,
            source,
            target,
            operand,
        } => {
            let operand = normalize_ranked_expression_v1(operand, lowering, ranked, next, budget)?;
            if source == target {
                operand
            } else {
                NormalizedScalarExpressionV1::Cast {
                    kind: *kind,
                    source: *source,
                    target: *target,
                    operand: Box::new(operand),
                }
            }
        }
    })
}

fn normalize_kir_expression_v1(
    function: &Function,
    kir: &KirCorrelationIndexV1<'_>,
    semantic_sites: &BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1>,
    value: ValueId,
    depth: usize,
    visiting: &mut BTreeSet<ValueId>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<NormalizedScalarExpressionV1> {
    budget.charge()?;
    if depth > MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 || !visiting.insert(value) {
        return None;
    }
    let result = normalize_kir_expression_inner_v1(
        function,
        kir,
        semantic_sites,
        value,
        depth,
        visiting,
        budget,
    );
    visiting.remove(&value);
    result
}

fn normalize_kir_expression_inner_v1(
    function: &Function,
    kir: &KirCorrelationIndexV1<'_>,
    semantic_sites: &BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1>,
    value: ValueId,
    depth: usize,
    visiting: &mut BTreeSet<ValueId>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<NormalizedScalarExpressionV1> {
    let body = function.body.as_ref()?;
    if let Some(parameter) = body
        .parameters
        .iter()
        .position(|candidate| *candidate == value)
    {
        let scalar = kir_semantic_scalar_v1(function.signature.parameters.get(parameter)?)?;
        let argument = u32::try_from(parameter).ok()?;
        let symbol = PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2.checked_add(argument)?;
        return Some(NormalizedScalarExpressionV1::Symbol { symbol, scalar });
    }
    if body
        .blocks
        .iter()
        .flat_map(|block| &block.parameters)
        .any(|parameter| parameter.id == value)
    {
        return None;
    }
    let operation = kir.definitions.get(&value)?;
    let scalar = operation
        .results
        .iter()
        .find(|result| result.id == value)
        .and_then(|result| kir_semantic_scalar_v1(&result.ty))?;
    let next = depth.checked_add(1)?;
    let recurse = |operand,
                   visiting: &mut BTreeSet<ValueId>,
                   budget: &mut UnsupportedIndexCorrelationBudgetV1| {
        normalize_kir_expression_v1(
            function,
            kir,
            semantic_sites,
            operand,
            next,
            visiting,
            budget,
        )
    };
    Some(match &operation.kind {
        OperationKind::Constant(constant) => {
            let (constant_scalar, bits) = normalize_kir_constant_v1(constant)?;
            if constant_scalar != scalar {
                return None;
            }
            NormalizedScalarExpressionV1::Constant { scalar, bits }
        }
        OperationKind::Unary { op, operand } => NormalizedScalarExpressionV1::Unary {
            operation: match op {
                UnaryOp::Not => ProductionSemanticUnaryOpV2::Not,
                UnaryOp::Negate => ProductionSemanticUnaryOpV2::Negate,
            },
            scalar,
            operand: Box::new(recurse(*operand, visiting, budget)?),
        },
        OperationKind::Binary { op, lhs, rhs } => {
            let (operation, overflow) = normalize_kir_binary_v1(*op, operation, value)?;
            NormalizedScalarExpressionV1::Binary {
                operation,
                scalar,
                overflow,
                lhs: Box::new(recurse(*lhs, visiting, budget)?),
                rhs: Box::new(recurse(*rhs, visiting, budget)?),
            }
        }
        OperationKind::Compare {
            predicate,
            lhs,
            rhs,
        } => {
            let lhs_scalar = kir_value_scalar_v1(function, kir, *lhs)?;
            NormalizedScalarExpressionV1::Compare {
                operation: normalize_kir_comparison_v1(*predicate),
                operand_scalar: lhs_scalar,
                lhs: Box::new(recurse(*lhs, visiting, budget)?),
                rhs: Box::new(recurse(*rhs, visiting, budget)?),
            }
        }
        OperationKind::Select {
            condition,
            true_value,
            false_value,
        } => NormalizedScalarExpressionV1::Select {
            scalar,
            condition: Box::new(recurse(*condition, visiting, budget)?),
            when_true: Box::new(recurse(*true_value, visiting, budget)?),
            when_false: Box::new(recurse(*false_value, visiting, budget)?),
        },
        OperationKind::Cast { kind, value, to } => {
            let source = kir_value_scalar_v1(function, kir, *value)?;
            let target = kir_semantic_scalar_v1(to)?;
            let operand = recurse(*value, visiting, budget)?;
            if source == target {
                operand
            } else {
                NormalizedScalarExpressionV1::Cast {
                    kind: normalize_kir_cast_v1(*kind, source, target)?,
                    source,
                    target,
                    operand: Box::new(operand),
                }
            }
        }
        OperationKind::Load { .. } => {
            let location = *kir.definition_locations.get(&value)?;
            let site = *semantic_sites.get(&(location, 0))?;
            NormalizedScalarExpressionV1::Load { site, scalar }
        }
        _ => return None,
    })
}

fn kir_value_scalar_v1(
    function: &Function,
    kir: &KirCorrelationIndexV1<'_>,
    value: ValueId,
) -> Option<ProductionSemanticScalarTypeV2> {
    let body = function.body.as_ref()?;
    if let Some(parameter) = body
        .parameters
        .iter()
        .position(|candidate| *candidate == value)
    {
        return kir_semantic_scalar_v1(function.signature.parameters.get(parameter)?);
    }
    if let Some(parameter) = body
        .blocks
        .iter()
        .flat_map(|block| &block.parameters)
        .find(|parameter| parameter.id == value)
    {
        return kir_semantic_scalar_v1(&parameter.ty);
    }
    kir.definitions
        .get(&value)?
        .results
        .iter()
        .find(|result| result.id == value)
        .and_then(|result| kir_semantic_scalar_v1(&result.ty))
}

fn kir_semantic_scalar_v1(ty: &Type) -> Option<ProductionSemanticScalarTypeV2> {
    let Type::Scalar(scalar) = ty else {
        return None;
    };
    Some(match scalar {
        ScalarType::Bool => ProductionSemanticScalarTypeV2::Bool,
        ScalarType::I8 => ProductionSemanticScalarTypeV2::Integer {
            signed: true,
            bits: 8,
        },
        ScalarType::I16 => ProductionSemanticScalarTypeV2::Integer {
            signed: true,
            bits: 16,
        },
        ScalarType::I32 => ProductionSemanticScalarTypeV2::Integer {
            signed: true,
            bits: 32,
        },
        ScalarType::I64 => ProductionSemanticScalarTypeV2::Integer {
            signed: true,
            bits: 64,
        },
        ScalarType::U8 => ProductionSemanticScalarTypeV2::Integer {
            signed: false,
            bits: 8,
        },
        ScalarType::U16 => ProductionSemanticScalarTypeV2::Integer {
            signed: false,
            bits: 16,
        },
        ScalarType::U32 => ProductionSemanticScalarTypeV2::Integer {
            signed: false,
            bits: 32,
        },
        ScalarType::U64 | ScalarType::Index => ProductionSemanticScalarTypeV2::Integer {
            signed: false,
            bits: 64,
        },
        ScalarType::F32 => ProductionSemanticScalarTypeV2::Float { bits: 32 },
        ScalarType::F64 => ProductionSemanticScalarTypeV2::Float { bits: 64 },
        ScalarType::I128 | ScalarType::U128 | ScalarType::F16 | ScalarType::Bf16 => return None,
    })
}

fn normalize_kir_constant_v1(constant: &Constant) -> Option<(ProductionSemanticScalarTypeV2, u64)> {
    let bits = match constant {
        Constant::Bool(value) => u64::from(*value),
        Constant::I8(value) => u64::from(*value as u8),
        Constant::I16(value) => u64::from(*value as u16),
        Constant::I32(value) => u64::from(*value as u32),
        Constant::I64(value) => *value as u64,
        Constant::U8(value) => u64::from(*value),
        Constant::U16(value) => u64::from(*value),
        Constant::U32(value) => u64::from(*value),
        Constant::U64(value) | Constant::Index(value) | Constant::F64Bits(value) => *value,
        Constant::F32Bits(value) => u64::from(*value),
        Constant::F16Bits(_) | Constant::Bf16Bits(_) => return None,
    };
    Some((kir_semantic_scalar_v1(&constant.ty())?, bits))
}

fn normalize_kir_binary_v1(
    operation: BinaryOp,
    definition: &Operation,
    value: ValueId,
) -> Option<(ProductionSemanticBinaryOpV2, ProductionOverflowContractV2)> {
    let (operation, overflow) = match operation {
        BinaryOp::Add => (
            ProductionSemanticBinaryOpV2::Add,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::Subtract => (
            ProductionSemanticBinaryOpV2::Subtract,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::Multiply => (
            ProductionSemanticBinaryOpV2::Multiply,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::Divide => (
            ProductionSemanticBinaryOpV2::Divide,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::Remainder => (
            ProductionSemanticBinaryOpV2::Remainder,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::BitAnd => (
            ProductionSemanticBinaryOpV2::BitAnd,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::BitOr => (
            ProductionSemanticBinaryOpV2::BitOr,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::BitXor => (
            ProductionSemanticBinaryOpV2::BitXor,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::ShiftLeft => (
            ProductionSemanticBinaryOpV2::ShiftLeft,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::ShiftRight => (
            ProductionSemanticBinaryOpV2::ShiftRight,
            ProductionOverflowContractV2::Wrapping,
        ),
        BinaryOp::Checked(operation) => {
            if definition.results.first().map(|result| result.id) != Some(value) {
                return None;
            }
            let operation = match operation {
                CheckedBinaryOperator::Add => ProductionSemanticBinaryOpV2::Add,
                CheckedBinaryOperator::Subtract => ProductionSemanticBinaryOpV2::Subtract,
                CheckedBinaryOperator::Multiply => ProductionSemanticBinaryOpV2::Multiply,
            };
            (operation, ProductionOverflowContractV2::Checked)
        }
    };
    Some((operation, overflow))
}

const fn normalize_kir_comparison_v1(
    predicate: ComparePredicate,
) -> ProductionSemanticComparisonV2 {
    match predicate {
        ComparePredicate::Equal => ProductionSemanticComparisonV2::Equal,
        ComparePredicate::NotEqual => ProductionSemanticComparisonV2::NotEqual,
        ComparePredicate::LessThan => ProductionSemanticComparisonV2::LessThan,
        ComparePredicate::LessThanOrEqual => ProductionSemanticComparisonV2::LessOrEqual,
        ComparePredicate::GreaterThan => ProductionSemanticComparisonV2::GreaterThan,
        ComparePredicate::GreaterThanOrEqual => ProductionSemanticComparisonV2::GreaterOrEqual,
    }
}

fn normalize_kir_cast_v1(
    kind: CastKind,
    source: ProductionSemanticScalarTypeV2,
    target: ProductionSemanticScalarTypeV2,
) -> Option<ProductionSemanticCastV2> {
    match kind {
        CastKind::Truncate | CastKind::ZeroExtend | CastKind::SignExtend | CastKind::Bitcast
            if matches!(
                (source, target),
                (
                    ProductionSemanticScalarTypeV2::Bool
                        | ProductionSemanticScalarTypeV2::Integer { .. },
                    ProductionSemanticScalarTypeV2::Bool
                        | ProductionSemanticScalarTypeV2::Integer { .. }
                )
            ) =>
        {
            Some(ProductionSemanticCastV2::Integer)
        }
        CastKind::FloatExtend | CastKind::FloatTruncate | CastKind::Bitcast
            if source.is_float() && target.is_float() =>
        {
            Some(ProductionSemanticCastV2::FloatToFloat)
        }
        CastKind::IntegerToFloat if source.is_integer() && target.is_float() => {
            Some(ProductionSemanticCastV2::IntegerToFloat)
        }
        CastKind::FloatToInteger if source.is_float() && target.is_integer() => {
            Some(ProductionSemanticCastV2::FloatToIntegerSaturating)
        }
        _ => None,
    }
}

fn expected_gfx950_workgroup_allocation_identity_v1(
    operation: &Operation,
    operation_access_ordinal: u32,
) -> Option<(u64, u64)> {
    let format = match &operation.kind {
        OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
            kind: Gfx950LdsTransposeOperationKindV1::Stage { format, .. },
            ..
        }) if operation_access_ordinal == 1 => *format,
        OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
            kind: Gfx950LdsTransposeOperationKindV1::Read { format, .. },
            ..
        }) if operation_access_ordinal == 0 => *format,
        _ => return None,
    };
    Some(match format {
        Gfx950LdsTransposeFormatV1::Fp4E2M1 => (
            dialect_kernel::GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            dialect_kernel::GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        ),
        Gfx950LdsTransposeFormatV1::Fp8E4M3 => (
            dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
            dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
        ),
    })
}

fn validate_mir_pliron_translation_v1(
    module: &Module,
    correspondence: &SemanticKirCorrespondenceV1,
    lowering: &ProductionRankedKernelLoweringInputV1,
    sources: &[ProductionRankedAccessSourceV1],
    max_operations: usize,
) -> Result<ProductionMirPlironTranslationValidationV1, ProductionMirPlironTranslationErrorV1> {
    let [kernel] = module.kernels.as_slice() else {
        return Err(ProductionMirPlironTranslationErrorV1::KernelShape);
    };
    let Some(function) = module
        .functions
        .iter()
        .find(|function| function.id == kernel.entry)
    else {
        return Err(ProductionMirPlironTranslationErrorV1::KernelShape);
    };
    let Some(body) = function.body.as_ref() else {
        return Err(ProductionMirPlironTranslationErrorV1::KernelShape);
    };
    let work_limit = max_operations
        .checked_mul(UNSUPPORTED_INDEX_CORRELATION_STEPS_PER_OPERATION_V1)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let mut budget = UnsupportedIndexCorrelationBudgetV1 {
        remaining: work_limit,
    };
    let kir = build_kir_correlation_index(body, max_operations, &mut budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let semantic_sites = index_semantic_access_sites(correspondence, &kir, &mut budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let ranked = index_ranked_correlation(lowering, sources, max_operations, &mut budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    if let Some(location) = kir.unmodeled_memory_effects.first().copied() {
        return Err(
            ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect { location },
        );
    }

    let mut used_ranked_locations = BTreeSet::new();
    let mut effect_locations = Vec::new();
    let mut memory_effects = 0_usize;
    let mut value_expressions = 0_usize;
    for consumer in &kir.memory_consumers {
        budget
            .charge()
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
        if consumer.memory_space == dialect_kernel::MemorySpaceAttr::Private
            && compiler_owned_enum_payload_access_v1(
                consumer.pointer,
                &kir,
                correspondence,
                &mut BTreeSet::new(),
                &mut budget,
            )
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?
        {
            continue;
        }
        let Some(site) = semantic_sites
            .get(&(consumer.location, consumer.operation_access_ordinal))
            .copied()
        else {
            if consumer.memory_space == dialect_kernel::MemorySpaceAttr::Private {
                continue;
            }
            return Err(
                ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect {
                    location: consumer.location,
                },
            );
        };
        let (logical_site, source) = ranked_source_for_semantic_effect_v1(&ranked, site).ok_or(
            ProductionMirPlironTranslationErrorV1::MissingRankedEffect {
                semantic_block: site.block,
                semantic_statement: site.statement,
                semantic_access_ordinal: site.ordinal,
            },
        )?;
        if source.access != consumer.access {
            return Err(ProductionMirPlironTranslationErrorV1::AccessKindMismatch {
                location: consumer.location,
            });
        }
        if source.atomic != consumer.atomic {
            return Err(
                ProductionMirPlironTranslationErrorV1::AtomicContractMismatch {
                    location: consumer.location,
                },
            );
        }
        let definition = match source.allocation {
            IndexedRankedAllocationV1::View(ProductionRankedValueV1::Local(view)) => {
                ranked.view_definitions.get(&view).copied()
            }
            IndexedRankedAllocationV1::View(_) => None,
            IndexedRankedAllocationV1::Direct(definition) => Some(definition),
        };
        let definition = definition.ok_or(
            ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                location: consumer.location,
            },
        )?;
        if definition.memory_space != consumer.memory_space {
            return Err(ProductionMirPlironTranslationErrorV1::MemorySpaceMismatch {
                location: consumer.location,
            });
        }
        if consumer.memory_space == dialect_kernel::MemorySpaceAttr::Workgroup {
            let operation = kir.operations.get(&consumer.location).ok_or(
                ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                    location: consumer.location,
                },
            )?;
            let expected = expected_gfx950_workgroup_allocation_identity_v1(
                operation,
                consumer.operation_access_ordinal,
            );
            match (source.allocation, expected) {
                (IndexedRankedAllocationV1::View(_), None) => {}
                (IndexedRankedAllocationV1::Direct(definition), Some((origin, class)))
                    if definition.allocation_origin == origin
                        && definition.noalias_class == class => {}
                _ => {
                    return Err(
                        ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                            location: consumer.location,
                        },
                    );
                }
            }
        }
        if consumer.memory_space == dialect_kernel::MemorySpaceAttr::Global {
            let parameter = external_allocation_parameter_v1(
                function,
                &kir,
                consumer.pointer,
                &mut BTreeSet::new(),
                &mut budget,
            )
            .ok_or(
                ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                    location: consumer.location,
                },
            )?;
            let expected_origin = u64::from(parameter).checked_add(1).ok_or(
                ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                    location: consumer.location,
                },
            )?;
            if definition.allocation_origin != expected_origin {
                return Err(
                    ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                        location: consumer.location,
                    },
                );
            }
        }
        let first_logical_use =
            used_ranked_locations.insert((source.ranked_block, source.ranked_operation));
        if !first_logical_use && !matches!(source.allocation, IndexedRankedAllocationV1::Direct(_))
        {
            return Err(ProductionMirPlironTranslationErrorV1::ExtraRankedEffect {
                ranked_block: source.ranked_block,
                ranked_operation: source.ranked_operation,
            });
        }
        if let Some(ranked_value) = source.value {
            let operation = kir.operations.get(&consumer.location).ok_or(
                ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                    location: consumer.location,
                },
            )?;
            let executable_value = kir_written_value_v1(operation).ok_or(
                ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                    location: consumer.location,
                },
            )?;
            let ProductionRankedValueV1::Local(ranked_value) = ranked_value else {
                return Err(
                    ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                        location: consumer.location,
                    },
                );
            };
            let (ranked_expression, numerical_contract) =
                ranked.semantic_expressions.get(&ranked_value).ok_or(
                    ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                        location: consumer.location,
                    },
                )?;
            if *numerical_contract
                != ProductionNumericalContractV2::exact_for_expression(ranked_expression)
            {
                return Err(
                    ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                        location: consumer.location,
                    },
                );
            }
            let expected = normalize_ranked_expression_v1(
                ranked_expression,
                lowering,
                &ranked,
                0,
                &mut budget,
            )
            .ok_or(
                ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                    location: consumer.location,
                },
            )?;
            let actual = normalize_kir_expression_v1(
                function,
                &kir,
                &semantic_sites,
                executable_value,
                0,
                &mut BTreeSet::new(),
                &mut budget,
            )
            .ok_or(
                ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                    location: consumer.location,
                },
            )?;
            if actual != expected {
                return Err(
                    ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                        location: consumer.location,
                    },
                );
            }
            value_expressions = value_expressions
                .checked_add(1)
                .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
        }
        if first_logical_use {
            effect_locations.push((
                logical_site,
                consumer.location,
                consumer.operation_access_ordinal,
                (source.ranked_block, source.ranked_operation),
            ));
        }
        memory_effects = memory_effects
            .checked_add(1)
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    }
    for source in ranked.sources_by_site.values() {
        budget
            .charge()
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
        let is_private = match source.allocation {
            IndexedRankedAllocationV1::View(ProductionRankedValueV1::Local(view)) => ranked
                .view_definitions
                .get(&view)
                .is_some_and(|definition| {
                    definition.memory_space == dialect_kernel::MemorySpaceAttr::Private
                }),
            IndexedRankedAllocationV1::View(_) => false,
            IndexedRankedAllocationV1::Direct(definition) => {
                definition.memory_space == dialect_kernel::MemorySpaceAttr::Private
            }
        };
        if !used_ranked_locations.contains(&(source.ranked_block, source.ranked_operation))
            && !is_private
        {
            return Err(ProductionMirPlironTranslationErrorV1::ExtraRankedEffect {
                ranked_block: source.ranked_block,
                ranked_operation: source.ranked_operation,
            });
        }
    }
    validate_effect_control_flow_v1(body, lowering.kernel(), &effect_locations, &mut budget)?;

    let kir_synchronization = kir_synchronization_contracts_v1(body)?;
    let ranked_synchronization = ranked_synchronization_contracts_v1(lowering.kernel())?;
    if kir_synchronization != ranked_synchronization {
        return Err(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch);
    }
    let kir_tensors = kir_tensor_contracts_v1(body)?;
    let ranked_tensors = ranked_tensor_contracts_v1(lowering.kernel())?;
    if kir_tensors != ranked_tensors {
        return Err(ProductionMirPlironTranslationErrorV1::TensorContractMismatch);
    }
    let conservative_ranked_effects = lowering
        .kernel()
        .blocks()
        .iter()
        .flat_map(|block| block.operations())
        .filter(|operation| {
            matches!(
                operation,
                ProductionRankedOperationV1::AllocationEffect { .. }
            )
        })
        .count();
    Ok(ProductionMirPlironTranslationValidationV1 {
        semantic_sha256: *correspondence.semantic_sha256(),
        memory_effects,
        synchronization_effects: kir_synchronization.len(),
        tensor_operations: kir_tensors.len(),
        value_expressions,
        conservative_ranked_effects,
    })
}

fn ranked_source_for_semantic_effect_v1<'index>(
    ranked: &'index RankedCorrelationIndexV1,
    site: SemanticAccessSiteV1,
) -> Option<(SemanticAccessSiteV1, &'index IndexedRankedAccessSourceV1)> {
    if let Some(source) = ranked.sources_by_site.get(&site) {
        return Some((site, source));
    }
    let source = ranked
        .conservative_sources_by_statement
        .get(&(site.block, site.statement))?;
    let logical_site = ranked
        .sites_by_ranked_location
        .get(&(source.ranked_block, source.ranked_operation))
        .copied()?;
    Some((logical_site, source))
}

fn compiler_owned_enum_payload_access_v1(
    pointer: ValueId,
    kir: &KirCorrelationIndexV1<'_>,
    correspondence: &SemanticKirCorrespondenceV1,
    visiting: &mut BTreeSet<ValueId>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<bool> {
    budget.charge()?;
    if !visiting.insert(pointer) {
        return Some(false);
    }
    let Some(definition) = kir.definitions.get(&pointer) else {
        visiting.remove(&pointer);
        return Some(false);
    };
    let Some(location) = kir.definition_locations.get(&pointer).copied() else {
        visiting.remove(&pointer);
        return Some(false);
    };
    let synthetic_allocation = matches!(
        definition.kind,
        OperationKind::Alloca {
            address_space: AddressSpace::Private,
            ..
        }
    ) && correspondence
        .synthetic_operation_spans()
        .iter()
        .any(|span| {
            span.rule() == SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage
                && operation_span_contains_v1(
                    span.kernel_ir_block(),
                    span.first_operation_ordinal(),
                    span.operation_count(),
                    location,
                )
        });
    let result = if synthetic_allocation {
        true
    } else {
        match &definition.kind {
            OperationKind::GetElementPointer { base, .. } => {
                compiler_owned_enum_payload_access_v1(*base, kir, correspondence, visiting, budget)?
            }
            OperationKind::Cast { value, .. } => compiler_owned_enum_payload_access_v1(
                *value,
                kir,
                correspondence,
                visiting,
                budget,
            )?,
            OperationKind::Select {
                true_value,
                false_value,
                ..
            } => {
                compiler_owned_enum_payload_access_v1(
                    *true_value,
                    kir,
                    correspondence,
                    visiting,
                    budget,
                )? && compiler_owned_enum_payload_access_v1(
                    *false_value,
                    kir,
                    correspondence,
                    visiting,
                    budget,
                )?
            }
            _ => false,
        }
    };
    visiting.remove(&pointer);
    Some(result)
}

fn kir_written_value_v1(operation: &Operation) -> Option<ValueId> {
    match &operation.kind {
        OperationKind::Store { value, .. } => Some(*value),
        OperationKind::Atomic(atomic)
            if matches!(atomic.kind, AtomicKind::Store | AtomicKind::Exchange) =>
        {
            atomic.value
        }
        _ => None,
    }
}

#[derive(Debug, Eq, PartialEq)]
struct NormalizedEffectFlowV1 {
    entry_effects: BTreeSet<SemanticAccessSiteV1>,
    next_effects: BTreeSet<(SemanticAccessSiteV1, SemanticAccessSiteV1)>,
}

fn validate_effect_control_flow_v1(
    body: &FunctionBody,
    ranked: &fe2o3_pliron::ProductionRankedKernelV1,
    locations: &[(
        SemanticAccessSiteV1,
        FunctionOperationLocation,
        u32,
        (u32, u32),
    )],
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Result<(), ProductionMirPlironTranslationErrorV1> {
    let mut kir_events = BTreeMap::<u32, Vec<(u64, SemanticAccessSiteV1)>>::new();
    let mut ranked_events = BTreeMap::<u32, Vec<(u64, SemanticAccessSiteV1)>>::new();
    let mut seen_ranked_effects = BTreeMap::<(u32, u32), SemanticAccessSiteV1>::new();
    for (site, kir, operation_access_ordinal, ranked) in locations {
        budget
            .charge()
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
        if let Some(first) = seen_ranked_effects.get(ranked) {
            if (first.block, first.statement) != (site.block, site.statement) {
                return Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                    first_semantic_block: first.block,
                    first_semantic_statement: first.statement,
                    second_semantic_block: site.block,
                    second_semantic_statement: site.statement,
                });
            }
            continue;
        }
        seen_ranked_effects.insert(*ranked, *site);
        kir_events.entry(kir.block.0).or_default().push((
            (u64::try_from(kir.operation_index)
                .map_err(|_| ProductionMirPlironTranslationErrorV1::ResourceLimit)?
                << 32)
                | u64::from(*operation_access_ordinal),
            *site,
        ));
        ranked_events
            .entry(ranked.0)
            .or_default()
            .push((u64::from(ranked.1) << 32, *site));
    }
    for events in kir_events.values_mut().chain(ranked_events.values_mut()) {
        events.sort_unstable();
    }
    let kir_successors = body
        .blocks
        .iter()
        .map(|block| {
            let successors = kir_terminator_successors_v1(block.terminator.as_ref()?)?;
            Some((block.id.0, successors))
        })
        .collect::<Option<BTreeMap<_, _>>>()
        .ok_or(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
            first_semantic_block: 0,
            first_semantic_statement: None,
            second_semantic_block: 0,
            second_semantic_statement: None,
        })?;
    let ranked_successors = ranked
        .blocks()
        .iter()
        .enumerate()
        .map(|(block, contents)| {
            Some((
                u32::try_from(block).ok()?,
                ranked_terminator_successors_v1(contents.terminator()),
            ))
        })
        .collect::<Option<BTreeMap<_, _>>>()
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let kir_entry = body
        .blocks
        .first()
        .map(|block| block.id.0)
        .ok_or(ProductionMirPlironTranslationErrorV1::KernelShape)?;
    let kir_flow = effect_flow_signature_v1(kir_entry, &kir_events, &kir_successors, budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let ranked_flow = effect_flow_signature_v1(0, &ranked_events, &ranked_successors, budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    if kir_flow == ranked_flow {
        return Ok(());
    }
    let differing = kir_flow
        .next_effects
        .symmetric_difference(&ranked_flow.next_effects)
        .next()
        .copied()
        .or_else(|| {
            kir_flow
                .entry_effects
                .symmetric_difference(&ranked_flow.entry_effects)
                .next()
                .copied()
                .map(|site| (site, site))
        })
        .unwrap_or((
            SemanticAccessSiteV1 {
                block: 0,
                statement: None,
                ordinal: 0,
            },
            SemanticAccessSiteV1 {
                block: 0,
                statement: None,
                ordinal: 0,
            },
        ));
    Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
        first_semantic_block: differing.0.block,
        first_semantic_statement: differing.0.statement,
        second_semantic_block: differing.1.block,
        second_semantic_statement: differing.1.statement,
    })
}

fn effect_flow_signature_v1(
    entry: u32,
    events: &BTreeMap<u32, Vec<(u64, SemanticAccessSiteV1)>>,
    successors: &BTreeMap<u32, Vec<u32>>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<NormalizedEffectFlowV1> {
    if !successors.contains_key(&entry) {
        return None;
    }
    let entry_effects = first_reachable_effects_v1(entry, None, events, successors, budget)?;
    let mut next_effects = BTreeSet::new();
    for (&block, block_events) in events {
        for &(operation, site) in block_events {
            budget.charge()?;
            for next in
                first_reachable_effects_v1(block, Some(operation), events, successors, budget)?
            {
                budget.charge()?;
                next_effects.insert((site, next));
            }
        }
    }
    Some(NormalizedEffectFlowV1 {
        entry_effects,
        next_effects,
    })
}

fn first_reachable_effects_v1(
    block: u32,
    after_operation: Option<u64>,
    events: &BTreeMap<u32, Vec<(u64, SemanticAccessSiteV1)>>,
    successors: &BTreeMap<u32, Vec<u32>>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<BTreeSet<SemanticAccessSiteV1>> {
    budget.charge()?;
    if let Some((_, site)) = events.get(&block).and_then(|events| {
        events
            .iter()
            .find(|(operation, _)| after_operation.is_none_or(|after| *operation > after))
    }) {
        return Some(BTreeSet::from([*site]));
    }
    let mut found = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut pending = VecDeque::new();
    pending.extend(successors.get(&block)?.iter().copied());
    while let Some(current) = pending.pop_front() {
        budget.charge()?;
        if !visited.insert(current) {
            continue;
        }
        if let Some((_, site)) = events.get(&current).and_then(|events| events.first()) {
            found.insert(*site);
            continue;
        }
        pending.extend(successors.get(&current)?.iter().copied());
    }
    Some(found)
}

fn kir_terminator_successors_v1(terminator: &Terminator) -> Option<Vec<u32>> {
    let successors = match terminator {
        Terminator::Branch { target, .. } => vec![target.0],
        Terminator::ConditionalBranch {
            then_target,
            else_target,
            ..
        } => vec![then_target.0, else_target.0],
        Terminator::Switch {
            cases,
            default_target,
            ..
        } => cases
            .iter()
            .map(|case| case.target.0)
            .chain([default_target.0])
            .collect(),
        Terminator::IntegerSwitch {
            cases,
            default_target,
            ..
        } => cases
            .iter()
            .map(|case| case.target.0)
            .chain([default_target.0])
            .collect(),
        Terminator::Return { .. } | Terminator::Unreachable => Vec::new(),
    };
    Some(successors)
}

fn ranked_terminator_successors_v1(
    terminator: &fe2o3_pliron::ProductionRankedTerminatorV1,
) -> Vec<u32> {
    use fe2o3_pliron::ProductionRankedTerminatorV1 as RankedTerminator;
    match terminator {
        RankedTerminator::IndexLessThan {
            true_block,
            false_block,
            ..
        }
        | RankedTerminator::IndexLessThanArgs {
            true_block,
            false_block,
            ..
        }
        | RankedTerminator::IndexEqual {
            true_block,
            false_block,
            ..
        }
        | RankedTerminator::IndexEqualArgs {
            true_block,
            false_block,
            ..
        } => vec![*true_block, *false_block],
        RankedTerminator::AnalysisSplit {
            first_block,
            second_block,
            ..
        }
        | RankedTerminator::AnalysisSplitArgs {
            first_block,
            second_block,
            ..
        } => vec![*first_block, *second_block],
        RankedTerminator::Branch { target }
        | RankedTerminator::BranchArgs { target, .. }
        | RankedTerminator::BranchArgsAdd { target, .. }
        | RankedTerminator::BranchArgsAddAt { target, .. } => vec![*target],
        RankedTerminator::Return | RankedTerminator::Trap => Vec::new(),
    }
}

fn external_allocation_parameter_v1(
    function: &Function,
    kir: &KirCorrelationIndexV1<'_>,
    value: ValueId,
    visiting: &mut BTreeSet<ValueId>,
    budget: &mut UnsupportedIndexCorrelationBudgetV1,
) -> Option<u32> {
    budget.charge()?;
    if !visiting.insert(value) {
        return None;
    }
    let body = function.body.as_ref()?;
    if let Some(index) = body
        .parameters
        .iter()
        .position(|parameter| *parameter == value)
    {
        let ty = function.signature.parameters.get(index)?;
        visiting.remove(&value);
        return matches!(ty, Type::Pointer(_) | Type::Slice(_))
            .then(|| u32::try_from(index).ok())
            .flatten();
    }
    let operation = kir.definitions.get(&value)?;
    let result = match &operation.kind {
        OperationKind::SliceData { slice } => {
            external_allocation_parameter_v1(function, kir, *slice, visiting, budget)
        }
        OperationKind::GetElementPointer { base, .. } => {
            external_allocation_parameter_v1(function, kir, *base, visiting, budget)
        }
        OperationKind::Cast { value, .. } => {
            external_allocation_parameter_v1(function, kir, *value, visiting, budget)
        }
        OperationKind::Select {
            true_value,
            false_value,
            ..
        } => {
            let first =
                external_allocation_parameter_v1(function, kir, *true_value, visiting, budget);
            let second =
                external_allocation_parameter_v1(function, kir, *false_value, visiting, budget);
            (first == second).then_some(first).flatten()
        }
        _ => None,
    };
    visiting.remove(&value);
    result
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NormalizedSynchronizationV1 {
    execution_scope: Option<u8>,
    memory_scope: u8,
    ordering: u8,
    address_space: u8,
}

fn kir_synchronization_contracts_v1(
    body: &FunctionBody,
) -> Result<Vec<NormalizedSynchronizationV1>, ProductionMirPlironTranslationErrorV1> {
    let mut contracts = Vec::new();
    for operation in body.blocks.iter().flat_map(|block| &block.operations) {
        let contract = match &operation.kind {
            OperationKind::Barrier(barrier) => Some(NormalizedSynchronizationV1 {
                execution_scope: Some(normalize_kir_scope_v1(barrier.execution_scope)),
                memory_scope: normalize_kir_scope_v1(barrier.memory_scope),
                ordering: normalize_kir_order_v1(barrier.semantics.ordering)
                    .ok_or(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)?,
                address_space: singleton_kir_address_space_v1(&barrier.semantics.address_spaces)?,
            }),
            OperationKind::WorkgroupBarrier(barrier) => Some(NormalizedSynchronizationV1 {
                execution_scope: Some(normalize_kir_scope_v1(SynchronizationScope::Workgroup)),
                memory_scope: normalize_kir_scope_v1(barrier.memory_scope),
                ordering: normalize_kir_order_v1(barrier.semantics.ordering)
                    .ok_or(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)?,
                address_space: singleton_kir_address_space_v1(&barrier.semantics.address_spaces)?,
            }),
            OperationKind::Fence(fence) => Some(NormalizedSynchronizationV1 {
                execution_scope: None,
                memory_scope: normalize_kir_scope_v1(fence.memory_scope),
                ordering: normalize_kir_order_v1(fence.semantics.ordering)
                    .ok_or(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)?,
                address_space: singleton_kir_address_space_v1(&fence.semantics.address_spaces)?,
            }),
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Publish { .. },
                ..
            }) => Some(NormalizedSynchronizationV1 {
                execution_scope: Some(normalize_kir_scope_v1(SynchronizationScope::Workgroup)),
                memory_scope: normalize_kir_scope_v1(SynchronizationScope::Workgroup),
                ordering: normalize_kir_order_v1(MemoryOrdering::AcquireRelease)
                    .ok_or(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)?,
                address_space: normalize_kir_address_space_v1(AddressSpace::Workgroup),
            }),
            _ => None,
        };
        let executable_synchronizations = operation
            .memory_effects()
            .into_iter()
            .filter(|effect| {
                matches!(
                    effect,
                    MemoryEffect::Synchronize { .. } | MemoryEffect::Fence { .. }
                )
            })
            .count();
        if executable_synchronizations != usize::from(contract.is_some()) {
            return Err(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch);
        }
        contracts.extend(contract);
    }
    contracts.sort_unstable();
    Ok(contracts)
}

fn ranked_synchronization_contracts_v1(
    kernel: &fe2o3_pliron::ProductionRankedKernelV1,
) -> Result<Vec<NormalizedSynchronizationV1>, ProductionMirPlironTranslationErrorV1> {
    let mut contracts = Vec::new();
    for operation in kernel.blocks().iter().flat_map(|block| block.operations()) {
        let contract = match operation {
            ProductionRankedOperationV1::Barrier {
                execution_scope,
                memory_scope,
                address_space,
                order,
            } => Some(NormalizedSynchronizationV1 {
                execution_scope: Some(normalize_ranked_hierarchy_v1(*execution_scope)),
                memory_scope: normalize_ranked_memory_scope_v1(*memory_scope),
                ordering: normalize_ranked_order_v1(*order),
                address_space: normalize_ranked_address_space_v1(*address_space),
            }),
            ProductionRankedOperationV1::Fence {
                memory_scope,
                address_space,
                order,
            } => Some(NormalizedSynchronizationV1 {
                execution_scope: None,
                memory_scope: normalize_ranked_memory_scope_v1(*memory_scope),
                ordering: normalize_ranked_order_v1(*order),
                address_space: normalize_ranked_address_space_v1(*address_space),
            }),
            _ => None,
        };
        contracts.extend(contract);
    }
    contracts.sort_unstable();
    Ok(contracts)
}

fn normalize_kir_scope_v1(scope: SynchronizationScope) -> u8 {
    match scope {
        SynchronizationScope::Invocation => 0,
        SynchronizationScope::Subgroup => 1,
        SynchronizationScope::Workgroup => 2,
        SynchronizationScope::Device => 3,
        SynchronizationScope::System => 4,
    }
}

fn normalize_ranked_hierarchy_v1(scope: dialect_gpu::HierarchyAttr) -> u8 {
    match scope {
        dialect_gpu::HierarchyAttr::Lane => 0,
        dialect_gpu::HierarchyAttr::Subgroup => 1,
        dialect_gpu::HierarchyAttr::Workgroup => 2,
        dialect_gpu::HierarchyAttr::Grid => 3,
    }
}

fn normalize_ranked_memory_scope_v1(scope: dialect_gpu::MemoryScopeAttr) -> u8 {
    match scope {
        dialect_gpu::MemoryScopeAttr::Subgroup => 1,
        dialect_gpu::MemoryScopeAttr::Workgroup => 2,
        dialect_gpu::MemoryScopeAttr::Device => 3,
        dialect_gpu::MemoryScopeAttr::System => 4,
    }
}

fn normalize_kir_order_v1(order: MemoryOrdering) -> Option<u8> {
    match order {
        MemoryOrdering::Relaxed => None,
        MemoryOrdering::Acquire => Some(1),
        MemoryOrdering::Release => Some(2),
        MemoryOrdering::AcquireRelease => Some(3),
        MemoryOrdering::SequentiallyConsistent => Some(4),
    }
}

fn normalize_ranked_order_v1(order: dialect_gpu::MemoryOrderAttr) -> u8 {
    match order {
        dialect_gpu::MemoryOrderAttr::Acquire => 1,
        dialect_gpu::MemoryOrderAttr::Release => 2,
        dialect_gpu::MemoryOrderAttr::AcquireRelease => 3,
        dialect_gpu::MemoryOrderAttr::SequentiallyConsistent => 4,
    }
}

fn normalize_kir_address_space_v1(space: AddressSpace) -> u8 {
    match space {
        AddressSpace::Private => 0,
        AddressSpace::Workgroup => 1,
        AddressSpace::Global => 2,
        AddressSpace::Constant => 3,
        AddressSpace::Generic => 4,
    }
}

fn normalize_ranked_address_space_v1(space: dialect_gpu::AddressSpaceAttr) -> u8 {
    match space {
        dialect_gpu::AddressSpaceAttr::Private => 0,
        dialect_gpu::AddressSpaceAttr::Workgroup => 1,
        dialect_gpu::AddressSpaceAttr::Global => 2,
        dialect_gpu::AddressSpaceAttr::Constant => 3,
    }
}

fn singleton_kir_address_space_v1(
    spaces: &BTreeSet<AddressSpace>,
) -> Result<u8, ProductionMirPlironTranslationErrorV1> {
    if spaces.len() != 1 {
        return Err(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch);
    }
    let space = spaces
        .first()
        .copied()
        .ok_or(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)?;
    Ok(normalize_kir_address_space_v1(space))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NormalizedTensorV1 {
    contract: TensorLayoutContractV1,
    active_lanes: u32,
    convergence: u8,
}

fn kir_tensor_contracts_v1(
    body: &FunctionBody,
) -> Result<Vec<NormalizedTensorV1>, ProductionMirPlironTranslationErrorV1> {
    let mut contracts = Vec::new();
    for operation in body.blocks.iter().flat_map(|block| &block.operations) {
        let OperationKind::Matrix(matrix) = &operation.kind else {
            continue;
        };
        let Some(contract) = matrix.tensor_layout else {
            continue;
        };
        contracts.push(NormalizedTensorV1 {
            contract,
            active_lanes: matrix.active_lanes,
            convergence: normalize_kir_scope_v1(matrix.convergence.scope()),
        });
    }
    contracts.sort_unstable();
    Ok(contracts)
}

fn ranked_tensor_contracts_v1(
    kernel: &fe2o3_pliron::ProductionRankedKernelV1,
) -> Result<Vec<NormalizedTensorV1>, ProductionMirPlironTranslationErrorV1> {
    let mut contracts = Vec::new();
    for operation in kernel.blocks().iter().flat_map(|block| block.operations()) {
        let ProductionRankedOperationV1::TensorLayout {
            contract,
            convergence,
            active_lanes,
            ..
        } = operation
        else {
            continue;
        };
        let convergence = match convergence {
            dialect_kernel::TensorConvergenceAttr::UniformSubgroup => 1,
            dialect_kernel::TensorConvergenceAttr::UniformWorkgroup => 2,
            dialect_kernel::TensorConvergenceAttr::Divergent
            | dialect_kernel::TensorConvergenceAttr::Opaque => {
                return Err(ProductionMirPlironTranslationErrorV1::TensorContractMismatch);
            }
        };
        contracts.push(NormalizedTensorV1 {
            contract: *contract,
            active_lanes: *active_lanes,
            convergence,
        });
    }
    contracts.sort_unstable();
    Ok(contracts)
}

fn indexed_ranked_source_matches_allocation(
    ranked: &RankedCorrelationIndexV1,
    source: &IndexedRankedAccessSourceV1,
    expected_access: dialect_kernel::AccessKindAttr,
    expected_memory_space: dialect_kernel::MemorySpaceAttr,
    parameter_index: u32,
) -> bool {
    let definition = match source.allocation {
        IndexedRankedAllocationV1::View(ProductionRankedValueV1::Local(view)) => {
            ranked.view_definitions.get(&view).copied()
        }
        IndexedRankedAllocationV1::View(_) => None,
        IndexedRankedAllocationV1::Direct(definition) => Some(definition),
    };
    let Some(definition) = definition else {
        return false;
    };
    source.access == expected_access
        && definition.memory_space == expected_memory_space
        && u64::from(parameter_index)
            .checked_add(1)
            .is_some_and(|expected| definition.allocation_origin == expected)
}

// Keep hostile shared predicate graphs bounded independently of the lowering budget.
const GUARDED_ADDRESS_PROOF_STEPS_PER_OPERATION_V1: usize = 32;

#[derive(Clone, Copy)]
enum GuardedAddressDefinitionV1<'module> {
    Parameter,
    Operation(&'module Operation),
}

struct GuardedAddressProofBudgetV1 {
    remaining: usize,
}

impl GuardedAddressProofBudgetV1 {
    fn charge(&mut self) -> Result<(), ()> {
        self.remaining = self.remaining.checked_sub(1).ok_or(())?;
        Ok(())
    }
}

fn guarded_accesses_have_structural_bounds_result(
    module: &Module,
    guarded_locations: &[FunctionOperationLocation],
    max_operations: usize,
) -> Result<(), ProductionMemoryDischargeFailureV1> {
    let [kernel] = module.kernels.as_slice() else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "guarded proof requires exactly one kernel",
        ));
    };
    let Some(function) = module.function(&kernel.entry) else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "guarded proof cannot find the selected kernel entry",
        ));
    };
    let Some(body) = function.body.as_ref() else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "guarded proof selected a kernel declaration without a body",
        ));
    };

    let mut definitions = BTreeMap::new();
    for parameter in &body.parameters {
        if definitions
            .insert(*parameter, GuardedAddressDefinitionV1::Parameter)
            .is_some()
        {
            return Err(ProductionMemoryDischargeFailureV1::stage(
                "guarded proof found a duplicate function parameter definition",
            ));
        }
    }

    let mut actual = BTreeMap::new();
    let mut operation_count = 0_usize;
    for block in &body.blocks {
        for parameter in &block.parameters {
            if definitions
                .insert(parameter.id, GuardedAddressDefinitionV1::Parameter)
                .is_some()
            {
                return Err(ProductionMemoryDischargeFailureV1::stage(
                    "guarded proof found a duplicate block parameter definition",
                ));
            }
        }
        for (ordinal, operation) in block.operations.iter().enumerate() {
            operation_count = match operation_count.checked_add(1) {
                Some(count) if count <= max_operations => count,
                _ => {
                    return Err(ProductionMemoryDischargeFailureV1::stage(
                        "guarded proof exceeded the operation resource limit",
                    ));
                }
            };
            for result in &operation.results {
                if definitions
                    .insert(result.id, GuardedAddressDefinitionV1::Operation(operation))
                    .is_some()
                {
                    return Err(ProductionMemoryDischargeFailureV1::access(
                        FunctionOperationLocation::new(block.id, ordinal),
                        "guarded proof found a duplicate operation result definition",
                    ));
                }
            }
            if matches!(
                &operation.kind,
                OperationKind::GuardedLoad { access, .. }
                    if access.address_space != AddressSpace::Private
            ) {
                let location = FunctionOperationLocation::new(block.id, ordinal);
                if actual.insert(location, operation).is_some() {
                    return Err(ProductionMemoryDischargeFailureV1::access(
                        location,
                        "guarded proof found a duplicate guarded-load location",
                    ));
                }
            }
        }
    }

    let provided = guarded_locations.iter().copied().collect::<BTreeSet<_>>();
    if actual.is_empty()
        || provided.len() != guarded_locations.len()
        || actual.keys().copied().collect::<BTreeSet<_>>() != provided
    {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "formal guarded-load locations do not match retained Kernel IR",
        ));
    }

    let Some(proof_steps) =
        max_operations.checked_mul(GUARDED_ADDRESS_PROOF_STEPS_PER_OPERATION_V1)
    else {
        return Err(ProductionMemoryDischargeFailureV1::stage(
            "guarded proof resource budget overflowed",
        ));
    };
    let mut budget = GuardedAddressProofBudgetV1 {
        remaining: proof_steps,
    };
    for (location, operation) in actual {
        if guarded_load_has_structural_bound(operation, &definitions, &mut budget) {
            continue;
        }
        if let Some((_predicate, index, slice)) =
            guarded_load_bound_subject(operation, &definitions)
        {
            return Err(ProductionMemoryDischargeFailureV1::guarded_bound(
                location,
                index,
                slice,
                "guard predicate does not prove the selected index is in bounds",
            ));
        }
        return Err(ProductionMemoryDischargeFailureV1::access(
            location,
            "guarded load does not retain the required slice/index/zero-fallback structure",
        ));
    }
    Ok(())
}

#[cfg(test)]
fn guarded_accesses_have_structural_bounds(
    module: &Module,
    guarded_locations: &[FunctionOperationLocation],
    max_operations: usize,
) -> bool {
    guarded_accesses_have_structural_bounds_result(module, guarded_locations, max_operations)
        .is_ok()
}

fn guarded_load_has_structural_bound(
    operation: &Operation,
    definitions: &BTreeMap<ValueId, GuardedAddressDefinitionV1<'_>>,
    budget: &mut GuardedAddressProofBudgetV1,
) -> bool {
    let Some((predicate, index, slice)) = guarded_load_bound_subject(operation, definitions) else {
        return false;
    };
    let mut visiting = BTreeSet::new();
    let mut memo = BTreeMap::new();
    predicate_implies_slice_bound(
        predicate,
        index,
        slice,
        definitions,
        budget,
        &mut visiting,
        &mut memo,
    )
    .unwrap_or(false)
}

fn guarded_load_bound_subject(
    operation: &Operation,
    definitions: &BTreeMap<ValueId, GuardedAddressDefinitionV1<'_>>,
) -> Option<(ValueId, ValueId, ValueId)> {
    let OperationKind::GuardedLoad {
        pointer, predicate, ..
    } = &operation.kind
    else {
        return None;
    };
    let Some(OperationKind::GetElementPointer { base, offset }) =
        operation_definition(definitions, *pointer).map(|operation| &operation.kind)
    else {
        return None;
    };
    let Some(OperationKind::SliceData { slice }) =
        operation_definition(definitions, *base).map(|operation| &operation.kind)
    else {
        return None;
    };
    let Some(OperationKind::Select {
        condition,
        true_value: index,
        false_value,
    }) = operation_definition(definitions, *offset).map(|operation| &operation.kind)
    else {
        return None;
    };
    if *condition != *predicate
        || !matches!(
            operation_definition(definitions, *false_value).map(|operation| &operation.kind),
            Some(OperationKind::Constant(Constant::Index(0)))
        )
    {
        return None;
    }
    Some((*predicate, *index, *slice))
}

fn operation_definition<'module>(
    definitions: &BTreeMap<ValueId, GuardedAddressDefinitionV1<'module>>,
    value: ValueId,
) -> Option<&'module Operation> {
    match definitions.get(&value) {
        Some(GuardedAddressDefinitionV1::Operation(operation)) => Some(*operation),
        Some(GuardedAddressDefinitionV1::Parameter) | None => None,
    }
}

fn predicate_implies_slice_bound(
    predicate: ValueId,
    index: ValueId,
    slice: ValueId,
    definitions: &BTreeMap<ValueId, GuardedAddressDefinitionV1<'_>>,
    budget: &mut GuardedAddressProofBudgetV1,
    visiting: &mut BTreeSet<ValueId>,
    memo: &mut BTreeMap<ValueId, bool>,
) -> Result<bool, ()> {
    if let Some(proved) = memo.get(&predicate) {
        return Ok(*proved);
    }
    budget.charge()?;
    if !visiting.insert(predicate) {
        return Err(());
    }
    let result = match definitions.get(&predicate) {
        Some(GuardedAddressDefinitionV1::Parameter) => Ok(false),
        Some(GuardedAddressDefinitionV1::Operation(operation)) => match &operation.kind {
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs,
                rhs,
            } if *lhs == index => match definitions.get(rhs) {
                Some(GuardedAddressDefinitionV1::Operation(operation)) if matches!(operation.kind, OperationKind::SliceLength { slice: bound_slice } if bound_slice == slice) => {
                    Ok(true)
                }
                Some(_) => Ok(false),
                None => Err(()),
            },
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs,
                rhs,
            } => {
                // A true conjunction inherits a bound implied by either conjunct.
                let lhs = predicate_implies_slice_bound(
                    *lhs,
                    index,
                    slice,
                    definitions,
                    budget,
                    visiting,
                    memo,
                )?;
                let rhs = predicate_implies_slice_bound(
                    *rhs,
                    index,
                    slice,
                    definitions,
                    budget,
                    visiting,
                    memo,
                )?;
                Ok(lhs || rhs)
            }
            _ => Ok(false),
        },
        None => Err(()),
    };
    visiting.remove(&predicate);
    if let Ok(proved) = result {
        memo.insert(predicate, proved);
    }
    result
}

fn mandatory_generic_checks_are_clean(lowering: &ProductionRankedKernelLoweringInputV1) -> bool {
    lowering.all_mandatory_reports_are_clean()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedSemanticKirBlockCoverageV1 {
    semantic_function: SemanticFunctionIdV1,
    semantic_block: SemanticBlockIdV1,
    kernel_ir_block: BlockId,
    source_statement_count: u32,
}

fn validate_semantic_kir_correspondence(
    owner: &ProductionSemanticMirOwnerV1,
    module: &Module,
    correspondence: &SemanticKirCorrespondenceV1,
    has_runtime_assert: bool,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let semantic = owner.semantic();
    if correspondence.semantic_sha256 != *semantic.semantic_sha256().as_bytes()
        || correspondence.function_count != semantic.functions().len()
    {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    let selection = semantic
        .select_kernel_body_v1()
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let function = semantic
        .functions()
        .get(selection.body().index() as usize)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let runtime_assert_rule =
        has_runtime_assert.then_some(SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap);
    let order = semantic_cfg_reverse_postorder(function)
        .map_err(|_| ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let expected = order
        .into_iter()
        .map(|semantic_block| {
            let block_index = usize::try_from(semantic_block.index())
                .map_err(|_| ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            let source = function
                .blocks()
                .get(block_index)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            Ok(ExpectedSemanticKirBlockCoverageV1 {
                semantic_function: selection.body(),
                semantic_block,
                kernel_ir_block: BlockId(semantic_block.index()),
                source_statement_count: u32::try_from(source.statements().len())
                    .map_err(|_| ProductionSemanticKirErrorV1::CorrespondenceMismatch)?,
            })
        })
        .collect::<Result<Vec<_>, ProductionSemanticKirErrorV1>>()?;
    let mut function_bodies = module
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref());
    let target_body = function_bodies
        .next()
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let target_blocks = target_body.blocks.as_slice();
    if function_bodies.next().is_some() {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    if validate_operation_correspondence_layout(
        &expected,
        target_blocks,
        &correspondence.blocks,
        &correspondence.statement_operation_spans,
        &correspondence.terminator_operation_spans,
        &correspondence.synthetic_operation_spans,
        runtime_assert_rule,
    ) && validate_parameter_correspondence_v1(
        selection.body(),
        function,
        target_body,
        &correspondence.parameter_bindings,
    ) {
        Ok(())
    } else {
        Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch)
    }
}

fn validate_parameter_correspondence_v1(
    semantic_function: SemanticFunctionIdV1,
    function: &SemanticFunctionDeclV1,
    target: &FunctionBody,
    bindings: &[SemanticKirParameterBindingV1],
) -> bool {
    let semantic_argument_count = function
        .locals()
        .iter()
        .filter(|declaration| matches!(declaration.role(), SemanticLocalRoleV1::Argument(_)))
        .count();
    bindings.len() == semantic_argument_count
        && bindings.len() == target.parameters.len()
        && bindings.iter().zip(&target.parameters).enumerate().all(
            |(argument, (binding, parameter))| {
                let Ok(local) = usize::try_from(binding.semantic_local.index()) else {
                    return false;
                };
                matches!(
                    function.locals().get(local).map(|declaration| declaration.role()),
                    Some(SemanticLocalRoleV1::Argument(actual))
                        if usize::try_from(actual) == Ok(argument)
                ) && binding.semantic_function == semantic_function
                    && binding.kernel_ir_value == *parameter
            },
        )
}

fn validate_operation_correspondence_layout(
    expected: &[ExpectedSemanticKirBlockCoverageV1],
    target_blocks: &[BasicBlock],
    blocks: &[SemanticKirBlockCorrespondenceV1],
    statements: &[SemanticKirStatementOperationSpanV1],
    terminators: &[SemanticKirTerminatorOperationSpanV1],
    synthetic: &[SemanticKirSyntheticOperationSpanV1],
    runtime_assert_rule: Option<SemanticKirSyntheticOperationRuleV1>,
) -> bool {
    let runtime_assert_block_count = usize::from(runtime_assert_rule.is_some());
    let Some(expected_target_blocks) = expected.len().checked_add(runtime_assert_block_count)
    else {
        return false;
    };
    if blocks.len() != expected.len()
        || terminators.len() != expected.len()
        || target_blocks.len() != expected_target_blocks
    {
        return false;
    }

    let mut statement_index = 0_usize;
    let mut synthetic_index = 0_usize;
    for (block_index, expected_block) in expected.iter().enumerate() {
        let Some(target) = target_blocks.get(block_index) else {
            return false;
        };
        let expected_block_record = SemanticKirBlockCorrespondenceV1 {
            semantic_function: expected_block.semantic_function,
            semantic_block: expected_block.semantic_block,
            kernel_ir_block: expected_block.kernel_ir_block,
            source_statement_count: expected_block.source_statement_count,
        };
        if blocks.get(block_index) != Some(&expected_block_record)
            || target.id != expected_block.kernel_ir_block
            || target.terminator.is_none()
        {
            return false;
        }

        let mut next_operation = 0_usize;
        if let Some(span) = synthetic.get(synthetic_index)
            && span.rule == SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage
            && span.kernel_ir_block == expected_block.kernel_ir_block
        {
            if span.first_operation_ordinal != 0 || span.operation_count == 0 {
                return false;
            }
            let Some(end) = checked_operation_span_end(
                span.first_operation_ordinal,
                span.operation_count,
                target.operations.len(),
            ) else {
                return false;
            };
            if !target.operations[..end].iter().all(|operation| {
                matches!(
                    operation.kind,
                    OperationKind::Alloca {
                        address_space: AddressSpace::Private,
                        ..
                    } | OperationKind::Load {
                        access: MemoryAccess {
                            address_space: AddressSpace::Private,
                            ..
                        },
                        ..
                    }
                )
            }) {
                return false;
            }
            next_operation = end;
            synthetic_index += 1;
        }
        for statement_ordinal in 0..expected_block.source_statement_count {
            let Some(span) = statements.get(statement_index) else {
                return false;
            };
            if span.semantic_function != expected_block.semantic_function
                || span.semantic_block != expected_block.semantic_block
                || span.statement_ordinal != statement_ordinal
                || span.kernel_ir_block != expected_block.kernel_ir_block
                || usize::try_from(span.first_operation_ordinal) != Ok(next_operation)
            {
                return false;
            }
            let Some(end) = checked_operation_span_end(
                span.first_operation_ordinal,
                span.operation_count,
                target.operations.len(),
            ) else {
                return false;
            };
            next_operation = end;
            statement_index += 1;
        }

        let Some(terminator) = terminators.get(block_index) else {
            return false;
        };
        if terminator.semantic_function != expected_block.semantic_function
            || terminator.semantic_block != expected_block.semantic_block
            || terminator.kernel_ir_block != expected_block.kernel_ir_block
            || usize::try_from(terminator.first_operation_ordinal) != Ok(next_operation)
        {
            return false;
        }
        let Some(end) = checked_operation_span_end(
            terminator.first_operation_ordinal,
            terminator.operation_count,
            target.operations.len(),
        ) else {
            return false;
        };
        if end != target.operations.len() {
            return false;
        }
    }
    if statement_index != statements.len() {
        return false;
    }

    if let Some(runtime_assert_rule) = runtime_assert_rule {
        let Some(span) = synthetic.get(synthetic_index) else {
            return false;
        };
        let Some(target) = target_blocks.get(expected.len()) else {
            return false;
        };
        let canonical_trap = AmdGpuDiagnosticOperation::Trap.operation(None);
        if span.rule != runtime_assert_rule
            || span.kernel_ir_block != target.id
            || span.first_operation_ordinal != 0
            || span.operation_count != 1
            || target.operations.as_slice() != [canonical_trap.clone()]
            || !matches!(target.terminator.as_ref(), Some(Terminator::Unreachable))
        {
            return false;
        }
        synthetic_index += 1;
    }
    synthetic_index == synthetic.len()
}

fn checked_operation_span_end(first: u32, count: u32, operation_len: usize) -> Option<usize> {
    let first = usize::try_from(first).ok()?;
    let count = usize::try_from(count).ok()?;
    first.checked_add(count).filter(|end| *end <= operation_len)
}

fn measured_operation_span(
    first: usize,
    after: usize,
    block: BlockId,
    statement: Option<u32>,
) -> Result<(u32, u32), ProductionSemanticKirErrorV1> {
    let count = after.checked_sub(first).ok_or_else(|| {
        unsupported(
            0,
            Some(block.0),
            statement,
            "Kernel IR operation count moved backwards during lowering",
        )
    })?;
    Ok((
        u32::try_from(first).map_err(|_| {
            unsupported(
                0,
                Some(block.0),
                statement,
                "Kernel IR operation ordinal is too large",
            )
        })?,
        u32::try_from(count).map_err(|_| {
            unsupported(
                0,
                Some(block.0),
                statement,
                "Kernel IR operation span is too large",
            )
        })?,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticScalarDefinitionV1 {
    Assignment { block: usize, statement: usize },
    Call { block: usize },
}

struct InfallibleBoundsAssertAnalysisV1<'a> {
    types: &'a [SemanticTypeDeclV1],
    callables: &'a [SemanticCallableDeclV1],
    function: &'a SemanticFunctionDeclV1,
    required_workgroup: [u32; 3],
    definitions: Vec<Option<SemanticScalarDefinitionV1>>,
    definition_counts: Vec<u8>,
    address_escaped: Vec<bool>,
    successors: Vec<Vec<usize>>,
    reachable: Vec<bool>,
    dominance: BTreeMap<(usize, usize), bool>,
}

impl<'a> InfallibleBoundsAssertAnalysisV1<'a> {
    fn analyze(
        types: &'a [SemanticTypeDeclV1],
        callables: &'a [SemanticCallableDeclV1],
        function: &'a SemanticFunctionDeclV1,
        required_workgroup: [u32; 3],
    ) -> Result<BTreeSet<u32>, ProductionSemanticKirErrorV1> {
        let local_count = function.locals().len();
        let mut definitions = vec![None; local_count];
        let mut definition_counts = vec![0_u8; local_count];
        let mut address_escaped = vec![false; local_count];
        let mut successors = Vec::new();
        successors
            .try_reserve_exact(function.blocks().len())
            .map_err(|_| {
                unsupported(
                    0,
                    None,
                    None,
                    "infallible bounds proof CFG storage cannot be reserved",
                )
            })?;

        {
            let mut record_definition =
                |place: &SemanticPlaceV1, definition: SemanticScalarDefinitionV1| {
                    let Some(local) = semantic_definition_local_v1(place) else {
                        return;
                    };
                    let Some(count) = definition_counts.get_mut(local) else {
                        return;
                    };
                    *count = count.saturating_add(1);
                    if *count == 1 && place.projections().is_empty() {
                        definitions[local] = Some(definition);
                    } else {
                        definitions[local] = None;
                    }
                };

            for (block_index, block) in function.blocks().iter().enumerate() {
                for (statement_index, statement) in block.statements().iter().enumerate() {
                    match statement.kind() {
                        SemanticStatementKindV1::Assign(assignment) => {
                            record_definition(
                                assignment.destination(),
                                SemanticScalarDefinitionV1::Assignment {
                                    block: block_index,
                                    statement: statement_index,
                                },
                            );
                            if let SemanticRvalueKindV1::Borrow { place, .. }
                            | SemanticRvalueKindV1::AddressOf { place, .. } =
                                assignment.value().kind()
                                && let Some(local) = semantic_definition_local_v1(place)
                                && let Some(escaped) = address_escaped.get_mut(local)
                            {
                                *escaped = true;
                            }
                        }
                        SemanticStatementKindV1::Store(store) => record_definition(
                            store.destination(),
                            SemanticScalarDefinitionV1::Assignment {
                                block: block_index,
                                statement: statement_index,
                            },
                        ),
                        SemanticStatementKindV1::AtomicRmw(atomic) => {
                            record_definition(
                                atomic.destination(),
                                SemanticScalarDefinitionV1::Assignment {
                                    block: block_index,
                                    statement: statement_index,
                                },
                            );
                            record_definition(
                                atomic.address(),
                                SemanticScalarDefinitionV1::Assignment {
                                    block: block_index,
                                    statement: statement_index,
                                },
                            );
                        }
                        SemanticStatementKindV1::AtomicCompareExchange(atomic) => {
                            record_definition(
                                atomic.destination(),
                                SemanticScalarDefinitionV1::Assignment {
                                    block: block_index,
                                    statement: statement_index,
                                },
                            );
                            record_definition(
                                atomic.address(),
                                SemanticScalarDefinitionV1::Assignment {
                                    block: block_index,
                                    statement: statement_index,
                                },
                            );
                        }
                        SemanticStatementKindV1::SetDiscriminant { place, .. }
                        | SemanticStatementKindV1::Deinitialize(place) => record_definition(
                            place,
                            SemanticScalarDefinitionV1::Assignment {
                                block: block_index,
                                statement: statement_index,
                            },
                        ),
                        SemanticStatementKindV1::StorageLive(_)
                        | SemanticStatementKindV1::StorageDead(_)
                        | SemanticStatementKindV1::Assume(_)
                        | SemanticStatementKindV1::Nop => {}
                    }
                }
                if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
                    && let Some(destination) = call.destination()
                {
                    record_definition(
                        destination.place(),
                        SemanticScalarDefinitionV1::Call { block: block_index },
                    );
                }
                let mut block_successors = Vec::new();
                block
                    .terminator()
                    .kind()
                    .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                        let target = edge.target().index() as usize;
                        if target >= function.blocks().len() {
                            return Err(unsupported(
                                0,
                                Some(block_index as u32),
                                None,
                                "infallible bounds proof references a missing CFG successor",
                            ));
                        }
                        block_successors.push(target);
                        Ok(())
                    })?;
                block_successors.sort_unstable();
                block_successors.dedup();
                successors.push(block_successors);
            }
        }

        let entry = function.entry().index() as usize;
        let reachable = semantic_reachable_blocks_v1(&successors, entry, None)?;
        let mut analysis = Self {
            types,
            callables,
            function,
            required_workgroup,
            definitions,
            definition_counts,
            address_escaped,
            successors,
            reachable,
            dominance: BTreeMap::new(),
        };
        let mut proved = BTreeSet::new();
        for (block, source) in function.blocks().iter().enumerate() {
            if analysis.proves_bounds_assert(block, source.terminator().kind())? {
                proved.insert(block as u32);
            }
        }
        Ok(proved)
    }

    fn proves_bounds_assert(
        &mut self,
        block: usize,
        terminator: &SemanticTerminatorKindV1,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        let SemanticTerminatorKindV1::Assert {
            condition,
            expected: true,
            message: SemanticAssertMessageV1::BoundsCheck { length, index },
            unwind: SemanticUnwindActionV1::Unreachable,
            ..
        } = terminator
        else {
            return Ok(false);
        };
        let (Some(condition_local), Some(index_local), Some(length_local)) = (
            whole_semantic_operand_local_v1(condition),
            whole_semantic_operand_local_v1(index),
            whole_semantic_operand_local_v1(length),
        ) else {
            return Ok(false);
        };
        if !self.condition_is_exact_less_than(condition_local, index, length, block)? {
            return Ok(false);
        }
        let Some(index_range) =
            self.range_of_local(index_local, block, usize::MAX, &mut BTreeSet::new())?
        else {
            return Ok(false);
        };
        let Some(exact_length) = self.exact_dominating_switch_value(length_local, block)? else {
            return Ok(false);
        };
        Ok(index_range.maximum < u128::from(exact_length))
    }

    fn condition_is_exact_less_than(
        &mut self,
        condition_local: usize,
        index: &SemanticOperandV1,
        length: &SemanticOperandV1,
        use_block: usize,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        let Some(SemanticScalarDefinitionV1::Assignment { block, statement }) =
            self.stable_definition(condition_local)
        else {
            return Ok(false);
        };
        if block != use_block {
            return Ok(false);
        }
        let Some(source) = self
            .function
            .blocks()
            .get(block)
            .and_then(|block| block.statements().get(statement))
        else {
            return Ok(false);
        };
        let SemanticStatementKindV1::Assign(assignment) = source.kind() else {
            return Ok(false);
        };
        let SemanticRvalueKindV1::Binary {
            operation: SemanticBinaryOpV1::LessThan,
            left,
            right,
        } = assignment.value().kind()
        else {
            return Ok(false);
        };
        Ok(left == index && right == length)
    }

    fn exact_dominating_switch_value(
        &mut self,
        local: usize,
        use_block: usize,
    ) -> Result<Option<u64>, ProductionSemanticKirErrorV1> {
        let Some(length_root) =
            self.stable_slice_length_root(local, use_block, usize::MAX, &mut BTreeSet::new())?
        else {
            return Ok(None);
        };
        let mut exact = None;
        for (source_block, block) in self.function.blocks().iter().enumerate() {
            let SemanticTerminatorKindV1::SwitchInt {
                discriminant,
                targets,
            } = block.terminator().kind()
            else {
                continue;
            };
            let Some(discriminant_local) = whole_semantic_operand_local_v1(discriminant) else {
                continue;
            };
            if self.stable_slice_length_root(
                discriminant_local,
                source_block,
                block.statements().len(),
                &mut BTreeSet::new(),
            )? == Some(length_root)
            {
                if !self.definition_dominates_use(
                    self.stable_definition(discriminant_local)
                        .expect("checked stable length definition"),
                    source_block,
                    block.statements().len(),
                )? {
                    continue;
                }
                for target in targets.values() {
                    let Ok(value) = u64::try_from(target.value()) else {
                        continue;
                    };
                    let edge_target = target.edge().target().index() as usize;
                    if targets.otherwise().target().index() as usize == edge_target
                        || targets.values().iter().any(|other| {
                            other.value() != target.value()
                                && other.edge().target().index() as usize == edge_target
                        })
                        || !self.edge_dominates(source_block, edge_target, use_block)?
                    {
                        continue;
                    }
                    match exact {
                        None => exact = Some(value),
                        Some(previous) if previous == value => {}
                        Some(_) => return Ok(None),
                    }
                }
                continue;
            }

            let Some((comparison, value)) = self.exact_length_comparison(
                discriminant_local,
                length_root,
                source_block,
                block.statements().len(),
            )?
            else {
                continue;
            };
            for target in targets.values() {
                let Ok(boolean) = u8::try_from(target.value()) else {
                    continue;
                };
                if boolean > 1 {
                    continue;
                }
                let edge_target = target.edge().target().index() as usize;
                if targets.otherwise().target().index() as usize == edge_target
                    || targets.values().iter().any(|other| {
                        other.value() != target.value()
                            && other.edge().target().index() as usize == edge_target
                    })
                    || !comparison_establishes_equality_v1(comparison, boolean != 0)
                    || !self.edge_dominates(source_block, edge_target, use_block)?
                {
                    continue;
                }
                match exact {
                    None => exact = Some(value),
                    Some(previous) if previous == value => {}
                    Some(_) => return Ok(None),
                }
            }
            if let [target] = targets.values()
                && target.value() <= 1
                && target.edge().target() != targets.otherwise().target()
                && comparison_establishes_equality_v1(comparison, target.value() == 0)
                && self.edge_dominates(
                    source_block,
                    targets.otherwise().target().index() as usize,
                    use_block,
                )?
            {
                match exact {
                    None => exact = Some(value),
                    Some(previous) if previous == value => {}
                    Some(_) => return Ok(None),
                }
            }
        }
        Ok(exact)
    }

    fn stable_slice_length_root(
        &mut self,
        local: usize,
        use_block: usize,
        use_statement: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<Option<usize>, ProductionSemanticKirErrorV1> {
        if !visiting.insert(local) {
            return Ok(None);
        }
        let result = match self.stable_definition(local) {
            Some(definition @ SemanticScalarDefinitionV1::Assignment { block, statement })
                if self.definition_dominates_use(definition, use_block, use_statement)? =>
            {
                let SemanticStatementKindV1::Assign(assignment) =
                    self.function.blocks()[block].statements()[statement].kind()
                else {
                    visiting.remove(&local);
                    return Ok(None);
                };
                match assignment.value().kind() {
                    SemanticRvalueKindV1::Length(_) => Some(local),
                    SemanticRvalueKindV1::Unary {
                        operation: SemanticUnaryOpV1::PointerMetadata,
                        operand,
                    } if self.is_exact_slice_length_metadata(
                        operand,
                        assignment.value().result_type(),
                    ) =>
                    {
                        Some(local)
                    }
                    SemanticRvalueKindV1::Use(operand)
                        if operand.ty() == assignment.value().result_type() =>
                    {
                        match whole_semantic_operand_local_v1(operand) {
                            Some(source) => {
                                self.stable_slice_length_root(source, block, statement, visiting)?
                            }
                            None => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        visiting.remove(&local);
        Ok(result)
    }

    fn is_exact_slice_length_metadata(
        &self,
        operand: &SemanticOperandV1,
        result_type: SemanticTypeIdV1,
    ) -> bool {
        let Some(SemanticTypeShapeV1::Pointer(pointer)) = self
            .types
            .get(operand.ty().index() as usize)
            .map(SemanticTypeDeclV1::shape)
        else {
            return false;
        };
        pointer.kind() == SemanticPointerKindV1::Reference
            && pointer.metadata() == SemanticPointerMetadataV1::SliceLength
            && matches!(
                self.types
                    .get(pointer.pointee().index() as usize)
                    .map(SemanticTypeDeclV1::shape),
                Some(SemanticTypeShapeV1::Slice { .. })
            )
            && self.unsigned_bits(result_type) == Some(64)
    }

    fn exact_length_comparison(
        &mut self,
        comparison_local: usize,
        length_root: usize,
        use_block: usize,
        use_statement: usize,
    ) -> Result<Option<(SemanticBinaryOpV1, u64)>, ProductionSemanticKirErrorV1> {
        let Some(definition @ SemanticScalarDefinitionV1::Assignment { block, statement }) =
            self.stable_definition(comparison_local)
        else {
            return Ok(None);
        };
        if !self.definition_dominates_use(definition, use_block, use_statement)? {
            return Ok(None);
        }
        let SemanticStatementKindV1::Assign(assignment) =
            self.function.blocks()[block].statements()[statement].kind()
        else {
            return Ok(None);
        };
        let SemanticRvalueKindV1::Binary {
            operation,
            left,
            right,
        } = assignment.value().kind()
        else {
            return Ok(None);
        };
        if !matches!(
            operation,
            SemanticBinaryOpV1::Equal | SemanticBinaryOpV1::NotEqual
        ) || !matches!(
            self.types
                .get(assignment.value().result_type().index() as usize)
                .map(SemanticTypeDeclV1::shape),
            Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool))
        ) {
            return Ok(None);
        }
        let left_is_length = match whole_semantic_operand_local_v1(left) {
            Some(local) => {
                self.stable_slice_length_root(local, block, statement, &mut BTreeSet::new())?
                    == Some(length_root)
            }
            None => false,
        };
        let right_is_length = match whole_semantic_operand_local_v1(right) {
            Some(local) => {
                self.stable_slice_length_root(local, block, statement, &mut BTreeSet::new())?
                    == Some(length_root)
            }
            None => false,
        };
        let (length, constant) = match (left_is_length, right_is_length) {
            (true, false) => (left, right),
            (false, true) => (right, left),
            _ => return Ok(None),
        };
        let Some(length_root_ty) = self
            .function
            .locals()
            .get(length_root)
            .map(|local| local.ty())
        else {
            return Ok(None);
        };
        if length.ty() != length_root_ty {
            return Ok(None);
        }
        let SemanticOperandV1::Constant(constant) = constant else {
            return Ok(None);
        };
        let SemanticConstantValueV1::Scalar(value) = constant.value() else {
            return Ok(None);
        };
        let Some(bits) = self.unsigned_bits(length.ty()) else {
            return Ok(None);
        };
        if constant.ty() != length.ty()
            || u16::from(value.size_bytes()) * 8 != bits
            || (bits < 128 && value.bits() >= (1_u128 << bits))
        {
            return Ok(None);
        }
        Ok(u64::try_from(value.bits())
            .ok()
            .map(|value| (*operation, value)))
    }

    fn range_of_local(
        &mut self,
        local: usize,
        use_block: usize,
        use_statement: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<Option<UnsignedSemanticRangeV1>, ProductionSemanticKirErrorV1> {
        if !visiting.insert(local) {
            return Ok(None);
        }
        let result = match self.stable_definition(local) {
            Some(definition)
                if self.definition_dominates_use(definition, use_block, use_statement)? =>
            {
                match definition {
                    SemanticScalarDefinitionV1::Assignment { block, statement } => {
                        let SemanticStatementKindV1::Assign(assignment) =
                            self.function.blocks()[block].statements()[statement].kind()
                        else {
                            visiting.remove(&local);
                            return Ok(None);
                        };
                        match assignment.value().kind() {
                            SemanticRvalueKindV1::Use(operand) => {
                                self.range_of_operand(operand, block, statement, visiting)?
                            }
                            SemanticRvalueKindV1::Cast {
                                kind: SemanticCastKindV1::Integer,
                                operand,
                            } if self
                                .unsigned_bits(assignment.value().result_type())
                                .zip(self.unsigned_bits(operand.ty()))
                                .is_some_and(|(destination, source)| destination >= source) =>
                            {
                                self.range_of_operand(operand, block, statement, visiting)?
                            }
                            _ => None,
                        }
                    }
                    SemanticScalarDefinitionV1::Call { block } => {
                        let SemanticTerminatorKindV1::Call(call) =
                            self.function.blocks()[block].terminator().kind()
                        else {
                            visiting.remove(&local);
                            return Ok(None);
                        };
                        if !call.arguments().is_empty()
                            || call.destination().is_none_or(|destination| {
                                semantic_definition_local_v1(destination.place()) != Some(local)
                                    || !destination.place().projections().is_empty()
                            })
                        {
                            None
                        } else {
                            match self.callables.get(call.callee().index() as usize) {
                                Some(SemanticCallableDeclV1::CompilerIntrinsic {
                                    operation:
                                        SemanticCompilerIntrinsicOperationV1::ThreadIndex(axis),
                                    ..
                                }) => {
                                    let extent = match axis {
                                        SemanticAxisV1::X => self.required_workgroup[0],
                                        SemanticAxisV1::Y => self.required_workgroup[1],
                                        SemanticAxisV1::Z => self.required_workgroup[2],
                                    };
                                    match (
                                        extent.checked_sub(1),
                                        self.unsigned_bits(
                                            call.destination()
                                                .expect("checked destination")
                                                .place()
                                                .ty(),
                                        ),
                                    ) {
                                        (Some(maximum), Some(bits)) => {
                                            let maximum = u128::from(maximum);
                                            let representable_maximum = if bits == 128 {
                                                u128::MAX
                                            } else {
                                                (1_u128 << bits) - 1
                                            };
                                            (maximum <= representable_maximum)
                                                .then_some(UnsignedSemanticRangeV1 { maximum })
                                        }
                                        _ => None,
                                    }
                                }
                                _ => None,
                            }
                        }
                    }
                }
            }
            _ => None,
        };
        visiting.remove(&local);
        Ok(result)
    }

    fn range_of_operand(
        &mut self,
        operand: &SemanticOperandV1,
        use_block: usize,
        use_statement: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<Option<UnsignedSemanticRangeV1>, ProductionSemanticKirErrorV1> {
        match operand {
            SemanticOperandV1::Constant(constant) => {
                let SemanticConstantValueV1::Scalar(value) = constant.value() else {
                    return Ok(None);
                };
                let Some(bits) = self.unsigned_bits(constant.ty()) else {
                    return Ok(None);
                };
                if bits < 128 && value.bits() >= (1_u128 << bits) {
                    return Ok(None);
                }
                Ok(Some(UnsignedSemanticRangeV1 {
                    maximum: value.bits(),
                }))
            }
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => {
                if !place.projections().is_empty() {
                    return Ok(None);
                }
                self.range_of_local(
                    place.local().index() as usize,
                    use_block,
                    use_statement,
                    visiting,
                )
            }
        }
    }

    fn unsigned_bits(&self, ty: SemanticTypeIdV1) -> Option<u16> {
        match self.types.get(ty.index() as usize)?.shape() {
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool) => Some(1),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits,
            }) if (1..=128).contains(bits) => Some(*bits),
            _ => None,
        }
    }

    fn stable_definition(&self, local: usize) -> Option<SemanticScalarDefinitionV1> {
        if self.definition_counts.get(local).copied() != Some(1)
            || self.address_escaped.get(local).copied() != Some(false)
        {
            return None;
        }
        self.definitions.get(local).copied().flatten()
    }

    fn definition_dominates_use(
        &mut self,
        definition: SemanticScalarDefinitionV1,
        use_block: usize,
        use_statement: usize,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        match definition {
            SemanticScalarDefinitionV1::Assignment { block, statement } => {
                if block == use_block {
                    Ok(statement < use_statement)
                } else {
                    self.block_dominates(block, use_block)
                }
            }
            SemanticScalarDefinitionV1::Call { block } => {
                let SemanticTerminatorKindV1::Call(call) =
                    self.function.blocks()[block].terminator().kind()
                else {
                    return Ok(false);
                };
                let Some(target) = call
                    .destination()
                    .map(|destination| destination.edge().target().index() as usize)
                else {
                    return Ok(false);
                };
                if matches!(
                    call.unwind(),
                    SemanticUnwindActionV1::Cleanup(edge)
                        if edge.target().index() as usize == target
                ) {
                    return Ok(false);
                }
                self.edge_dominates(block, target, use_block)
            }
        }
    }

    fn block_dominates(
        &mut self,
        dominator: usize,
        block: usize,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        if let Some(result) = self.dominance.get(&(dominator, block)).copied() {
            return Ok(result);
        }
        if dominator >= self.successors.len() || block >= self.successors.len() {
            return Ok(false);
        }
        let result = if dominator == block {
            self.reachable.get(block).copied().unwrap_or(false)
        } else if !self.reachable.get(dominator).copied().unwrap_or(false)
            || !self.reachable.get(block).copied().unwrap_or(false)
        {
            false
        } else {
            !semantic_reachable_blocks_avoiding_node_v1(
                &self.successors,
                self.function.entry().index() as usize,
                dominator,
            )?
            .get(block)
            .copied()
            .unwrap_or(false)
        };
        self.dominance.insert((dominator, block), result);
        Ok(result)
    }

    fn edge_dominates(
        &self,
        source: usize,
        target: usize,
        block: usize,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        if source >= self.successors.len()
            || target >= self.successors.len()
            || block >= self.successors.len()
            || !self.reachable.get(source).copied().unwrap_or(false)
            || !self.reachable.get(block).copied().unwrap_or(false)
            || !self.successors[source].contains(&target)
        {
            return Ok(false);
        }
        Ok(!semantic_reachable_blocks_v1(
            &self.successors,
            self.function.entry().index() as usize,
            Some((source, target)),
        )?
        .get(block)
        .copied()
        .unwrap_or(false))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UnsignedSemanticRangeV1 {
    maximum: u128,
}

fn comparison_establishes_equality_v1(operation: SemanticBinaryOpV1, value: bool) -> bool {
    matches!(
        (operation, value),
        (SemanticBinaryOpV1::Equal, true) | (SemanticBinaryOpV1::NotEqual, false)
    )
}

fn semantic_definition_local_v1(place: &SemanticPlaceV1) -> Option<usize> {
    (!matches!(
        place
            .projections()
            .first()
            .map(|projection| projection.kind()),
        Some(SemanticProjectionKindV1::Dereference)
    ))
    .then_some(place.local().index() as usize)
}

fn whole_semantic_operand_local_v1(operand: &SemanticOperandV1) -> Option<usize> {
    match operand {
        SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)
            if place.projections().is_empty() =>
        {
            Some(place.local().index() as usize)
        }
        SemanticOperandV1::Copy(_)
        | SemanticOperandV1::Move(_)
        | SemanticOperandV1::Constant(_) => None,
    }
}

fn semantic_unsigned_integer_bits_v1(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Option<u16> {
    match types.get(ty.index() as usize)?.shape() {
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
            signed: false,
            bits,
        }) if (1..=128).contains(bits) => Some(*bits),
        _ => None,
    }
}

fn exact_unsigned_semantic_constant_v1(
    types: &[SemanticTypeDeclV1],
    operand: &SemanticOperandV1,
    expected_type: SemanticTypeIdV1,
) -> Option<u128> {
    let SemanticOperandV1::Constant(constant) = operand else {
        return None;
    };
    let SemanticConstantValueV1::Scalar(value) = constant.value() else {
        return None;
    };
    let bits = semantic_unsigned_integer_bits_v1(types, expected_type)?;
    if constant.ty() != expected_type || u16::from(value.size_bytes()) * 8 != bits {
        return None;
    }
    let value = value.bits();
    ((bits == 128) || value < (1_u128 << bits)).then_some(value)
}

fn semantic_unsigned_type_contains_exclusive_bound_v1(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    exclusive_bound: u128,
) -> bool {
    let Some(bits) = semantic_unsigned_integer_bits_v1(types, ty) else {
        return false;
    };
    exclusive_bound == 0 || bits == 128 || exclusive_bound - 1 < (1_u128 << bits)
}

fn resolve_semantic_header_copy_alias_v1(
    block: &fe2o3_mir_model::semantic_mir_v1::SemanticBasicBlockV1,
    before_statement: usize,
    operand: &SemanticOperandV1,
) -> Option<u32> {
    let mut current = u32::try_from(whole_semantic_operand_local_v1(operand)?).ok()?;
    let mut visited = BTreeSet::new();
    for _ in 0..=before_statement {
        if !visited.insert(current) {
            return None;
        }
        let definitions = block.statements()[..before_statement]
            .iter()
            .filter_map(|statement| {
                let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                    return None;
                };
                (semantic_definition_local_v1(assignment.destination())
                    == usize::try_from(current).ok())
                .then_some(assignment)
            })
            .collect::<Vec<_>>();
        if definitions.is_empty() {
            return Some(current);
        }
        if definitions.len() != 1 || !definitions[0].destination().projections().is_empty() {
            return None;
        }
        let SemanticRvalueKindV1::Use(source) = definitions[0].value().kind() else {
            return None;
        };
        if definitions[0].destination().ty() != source.ty() {
            return None;
        }
        current = u32::try_from(whole_semantic_operand_local_v1(source)?).ok()?;
    }
    None
}

fn semantic_cfg_graph_v1(
    function: &SemanticFunctionDeclV1,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<bool>), ProductionSemanticKirErrorV1> {
    let mut successors = vec![Vec::new(); function.blocks().len()];
    let mut predecessors = vec![Vec::new(); function.blocks().len()];
    for (source, block) in function.blocks().iter().enumerate() {
        block
            .terminator()
            .kind()
            .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                let target = edge.target().index() as usize;
                if target >= function.blocks().len() {
                    return Err(unsupported(
                        0,
                        Some(source as u32),
                        None,
                        "authenticated induction proof references a missing CFG successor",
                    ));
                }
                successors[source].push(target);
                Ok(())
            })?;
        successors[source].sort_unstable();
        successors[source].dedup();
        for target in successors[source].iter().copied() {
            predecessors[target].push(source);
        }
    }
    let reachable =
        semantic_reachable_blocks_v1(&successors, function.entry().index() as usize, None)?;
    Ok((successors, predecessors, reachable))
}

fn authenticated_natural_loop_topology_v1(
    function: &SemanticFunctionDeclV1,
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
    reachable: &[bool],
    header: usize,
    body_entry: usize,
    exit: usize,
) -> Option<(usize, usize, Vec<usize>)> {
    if !reachable.get(header).copied().unwrap_or(false)
        || body_entry >= successors.len()
        || exit >= successors.len()
        || body_entry == exit
    {
        return None;
    }
    let mut reachable_without_header = vec![false; successors.len()];
    let entry = function.entry().index() as usize;
    let mut pending = (entry != header)
        .then_some(entry)
        .into_iter()
        .collect::<Vec<_>>();
    while let Some(block) = pending.pop() {
        if block == header || reachable_without_header[block] {
            continue;
        }
        reachable_without_header[block] = true;
        pending.extend(successors[block].iter().copied());
    }
    let header_predecessors = predecessors[header]
        .iter()
        .copied()
        .filter(|predecessor| reachable[*predecessor])
        .collect::<Vec<_>>();
    let backedges = header_predecessors
        .iter()
        .copied()
        .filter(|predecessor| !reachable_without_header[*predecessor])
        .collect::<Vec<_>>();
    let preheaders = header_predecessors
        .iter()
        .copied()
        .filter(|predecessor| reachable_without_header[*predecessor])
        .collect::<Vec<_>>();
    if backedges.len() != 1 || preheaders.len() != 1 {
        return None;
    }
    let latch = backedges[0];
    let preheader = preheaders[0];
    if !matches!(
        function.blocks()[preheader].terminator().kind(),
        SemanticTerminatorKindV1::Goto(edge) if edge.target().index() as usize == header
    ) || !matches!(
        function.blocks()[latch].terminator().kind(),
        SemanticTerminatorKindV1::Goto(edge) if edge.target().index() as usize == header
    ) {
        return None;
    }

    let mut in_loop = vec![false; successors.len()];
    in_loop[header] = true;
    in_loop[latch] = true;
    let mut pending = vec![latch];
    while let Some(block) = pending.pop() {
        if block == header {
            continue;
        }
        for predecessor in predecessors[block].iter().copied() {
            if reachable[predecessor] && !in_loop[predecessor] {
                in_loop[predecessor] = true;
                pending.push(predecessor);
            }
        }
    }
    if !in_loop[body_entry] || in_loop[exit] {
        return None;
    }
    for (block, inside) in in_loop.iter().copied().enumerate() {
        if !inside || reachable_without_header[block] {
            if inside && reachable_without_header[block] {
                return None;
            }
            continue;
        }
        if predecessors[block].iter().copied().any(|predecessor| {
            reachable[predecessor]
                && !in_loop[predecessor]
                && !(block == header && predecessor == preheader)
        }) {
            return None;
        }
    }
    let mut exits = Vec::new();
    for (source, inside) in in_loop.iter().copied().enumerate() {
        if !inside {
            continue;
        }
        for target in successors[source].iter().copied() {
            if !in_loop[target] {
                exits.push((source, target));
            }
        }
    }
    if exits.as_slice() != [(header, exit)] {
        return None;
    }
    Some((
        preheader,
        latch,
        in_loop
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(block, inside)| inside.then_some(block))
            .collect(),
    ))
}

fn authenticated_loop_induction_shape_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    induction: u32,
    preheader: usize,
    latch: usize,
    exclusive_bound: u128,
) -> Option<()> {
    let induction_index = induction as usize;
    let induction_type = function.locals().get(induction_index)?.ty();
    let bits = semantic_unsigned_integer_bits_v1(types, induction_type)?;
    let mut initial = None;
    let mut step = None;
    let mut definitions = 0_u8;
    for (block_index, block) in function.blocks().iter().enumerate() {
        for statement in block.statements() {
            match statement.kind() {
                SemanticStatementKindV1::Assign(assignment) => {
                    if let SemanticRvalueKindV1::Borrow { place, .. }
                    | SemanticRvalueKindV1::AddressOf { place, .. } = assignment.value().kind()
                        && semantic_definition_local_v1(place) == Some(induction_index)
                    {
                        return None;
                    }
                    if semantic_definition_local_v1(assignment.destination())
                        != Some(induction_index)
                    {
                        continue;
                    }
                    if !assignment.destination().projections().is_empty() {
                        return None;
                    }
                    definitions = definitions.checked_add(1)?;
                    if block_index == preheader {
                        let SemanticRvalueKindV1::Use(value) = assignment.value().kind() else {
                            return None;
                        };
                        initial = Some(exact_unsigned_semantic_constant_v1(
                            types,
                            value,
                            induction_type,
                        )?);
                    } else if block_index == latch {
                        let SemanticRvalueKindV1::Binary {
                            operation: SemanticBinaryOpV1::Add,
                            left,
                            right,
                        } = assignment.value().kind()
                        else {
                            return None;
                        };
                        let constant = if whole_semantic_operand_local_v1(left)
                            == Some(induction_index)
                        {
                            right
                        } else if whole_semantic_operand_local_v1(right) == Some(induction_index) {
                            left
                        } else {
                            return None;
                        };
                        step = Some(exact_unsigned_semantic_constant_v1(
                            types,
                            constant,
                            induction_type,
                        )?);
                    } else {
                        return None;
                    }
                }
                SemanticStatementKindV1::Store(store)
                    if semantic_definition_local_v1(store.destination())
                        == Some(induction_index) =>
                {
                    return None;
                }
                SemanticStatementKindV1::AtomicRmw(atomic)
                    if semantic_definition_local_v1(atomic.destination())
                        == Some(induction_index)
                        || semantic_definition_local_v1(atomic.address())
                            == Some(induction_index) =>
                {
                    return None;
                }
                SemanticStatementKindV1::AtomicCompareExchange(atomic)
                    if semantic_definition_local_v1(atomic.destination())
                        == Some(induction_index)
                        || semantic_definition_local_v1(atomic.address())
                            == Some(induction_index) =>
                {
                    return None;
                }
                SemanticStatementKindV1::SetDiscriminant { place, .. }
                | SemanticStatementKindV1::Deinitialize(place)
                    if semantic_definition_local_v1(place) == Some(induction_index) =>
                {
                    return None;
                }
                _ => {}
            }
        }
        if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
            && call.destination().is_some_and(|destination| {
                semantic_definition_local_v1(destination.place()) == Some(induction_index)
            })
        {
            return None;
        }
    }
    let step = step?;
    if definitions != 2 || initial != Some(0) || step == 0 || exclusive_bound == 0 {
        return None;
    }
    let maximum = if bits == 128 {
        u128::MAX
    } else {
        (1_u128 << bits) - 1
    };
    (exclusive_bound - 1)
        .checked_add(step)
        .filter(|next| *next <= maximum)
        .map(|_| ())
}

fn authenticated_loop_induction_bounds_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
) -> Result<BTreeMap<(u32, u32), u128>, ProductionSemanticKirErrorV1> {
    let (successors, predecessors, reachable) = semantic_cfg_graph_v1(function)?;
    let mut bounds = BTreeMap::new();
    for (header, block) in function.blocks().iter().enumerate() {
        let SemanticTerminatorKindV1::SwitchInt {
            discriminant,
            targets,
        } = block.terminator().kind()
        else {
            continue;
        };
        if targets.values().len() != 1 || targets.values()[0].value() != 0 {
            continue;
        }
        let Some(discriminant_local) = whole_semantic_operand_local_v1(discriminant) else {
            continue;
        };
        let comparisons = block
            .statements()
            .iter()
            .enumerate()
            .filter_map(|(statement, source)| {
                let SemanticStatementKindV1::Assign(assignment) = source.kind() else {
                    return None;
                };
                (semantic_definition_local_v1(assignment.destination()) == Some(discriminant_local))
                    .then_some((statement, assignment))
            })
            .collect::<Vec<_>>();
        let [(comparison_statement, comparison)] = comparisons.as_slice() else {
            continue;
        };
        if !comparison.destination().projections().is_empty() {
            continue;
        }
        let SemanticRvalueKindV1::Binary {
            operation: SemanticBinaryOpV1::LessThan,
            left,
            right,
        } = comparison.value().kind()
        else {
            continue;
        };
        let Some(induction) =
            resolve_semantic_header_copy_alias_v1(block, *comparison_statement, left)
        else {
            continue;
        };
        let Some(induction_type) = function
            .locals()
            .get(induction as usize)
            .map(|local| local.ty())
        else {
            continue;
        };
        let Some(exclusive_bound) =
            exact_unsigned_semantic_constant_v1(types, right, induction_type)
        else {
            continue;
        };
        let body_entry = targets.otherwise().target().index() as usize;
        let exit = targets.values()[0].edge().target().index() as usize;
        let Some((preheader, latch, loop_blocks)) = authenticated_natural_loop_topology_v1(
            function,
            &successors,
            &predecessors,
            &reachable,
            header,
            body_entry,
            exit,
        ) else {
            continue;
        };
        if authenticated_loop_induction_shape_v1(
            types,
            function,
            induction,
            preheader,
            latch,
            exclusive_bound,
        )
        .is_none()
        {
            continue;
        }
        for body_block in loop_blocks.into_iter().filter(|body| *body != header) {
            bounds.insert((body_block as u32, induction), exclusive_bound);
        }
    }
    Ok(bounds)
}

fn authenticated_unsigned_operand_exclusive_bound_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    bounds: &BTreeMap<(u32, u32), u128>,
    block: SemanticBlockIdV1,
    operand: &SemanticOperandV1,
) -> Option<u128> {
    let source_block = function.blocks().get(block.index() as usize)?;
    let mut current = u32::try_from(whole_semantic_operand_local_v1(operand)?).ok()?;
    let mut traversed_types = vec![operand.ty()];
    let mut visited = BTreeSet::new();
    for _ in 0..=source_block.statements().len() {
        if !visited.insert(current) {
            return None;
        }
        if let Some(bound) = bounds.get(&(block.index(), current)).copied() {
            return traversed_types
                .iter()
                .all(|ty| semantic_unsigned_type_contains_exclusive_bound_v1(types, *ty, bound))
                .then_some(bound);
        }
        let definitions = source_block
            .statements()
            .iter()
            .filter_map(|statement| {
                let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                    return None;
                };
                (semantic_definition_local_v1(assignment.destination()) == Some(current as usize))
                    .then_some(assignment)
            })
            .collect::<Vec<_>>();
        let [definition] = definitions.as_slice() else {
            return None;
        };
        if !definition.destination().projections().is_empty()
            || definition.destination().ty() != *traversed_types.last()?
        {
            return None;
        }
        let source = match definition.value().kind() {
            SemanticRvalueKindV1::Use(source) => source,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Integer,
                operand: source,
            } => source,
            _ => return None,
        };
        current = u32::try_from(whole_semantic_operand_local_v1(source)?).ok()?;
        traversed_types.push(source.ty());
    }
    None
}

fn semantic_reachable_blocks_v1(
    successors: &[Vec<usize>],
    entry: usize,
    removed_edge: Option<(usize, usize)>,
) -> Result<Vec<bool>, ProductionSemanticKirErrorV1> {
    if entry >= successors.len() {
        return Err(unsupported(
            0,
            None,
            None,
            "infallible bounds proof entry block is missing",
        ));
    }
    let mut reachable = vec![false; successors.len()];
    let mut pending = VecDeque::from([entry]);
    reachable[entry] = true;
    while let Some(source) = pending.pop_front() {
        for target in &successors[source] {
            if removed_edge == Some((source, *target)) || reachable[*target] {
                continue;
            }
            reachable[*target] = true;
            pending.push_back(*target);
        }
    }
    Ok(reachable)
}

fn semantic_reachable_blocks_avoiding_node_v1(
    successors: &[Vec<usize>],
    entry: usize,
    removed_node: usize,
) -> Result<Vec<bool>, ProductionSemanticKirErrorV1> {
    if entry >= successors.len() || removed_node >= successors.len() {
        return Err(unsupported(
            0,
            None,
            None,
            "infallible bounds proof node query is outside the CFG",
        ));
    }
    let mut reachable = vec![false; successors.len()];
    if entry == removed_node {
        return Ok(reachable);
    }
    let mut pending = VecDeque::from([entry]);
    reachable[entry] = true;
    while let Some(source) = pending.pop_front() {
        for target in &successors[source] {
            if *target == removed_node || reachable[*target] {
                continue;
            }
            reachable[*target] = true;
            pending.push_back(*target);
        }
    }
    Ok(reachable)
}

fn semantic_requires_runtime_assert_failure(
    function: &SemanticFunctionDeclV1,
    infallible_asserts: &BTreeSet<u32>,
) -> bool {
    function
        .blocks()
        .iter()
        .enumerate()
        .any(|(block_index, block)| match block.terminator().kind() {
            SemanticTerminatorKindV1::Assert { .. } => {
                !infallible_asserts.contains(&(block_index as u32))
            }
            SemanticTerminatorKindV1::Abort | SemanticTerminatorKindV1::UnwindTerminate => true,
            _ => false,
        })
}

fn lower_module(
    owner: &ProductionSemanticMirOwnerV1,
    limits: ProductionSemanticKirLimitsV1,
    authenticated_launch_rank: Option<u8>,
) -> Result<(Module, SemanticKirCorrespondenceV1), ProductionSemanticKirErrorV1> {
    let semantic = owner.semantic();
    let launch_rank = authenticated_launch_rank.unwrap_or(1);
    if !(1..=3).contains(&launch_rank) {
        return Err(unsupported(
            0,
            None,
            None,
            "authenticated launch rank is outside the supported range",
        ));
    }
    enforce_limit(
        ProductionSemanticKirResourceV1::Functions,
        semantic.functions().len(),
        limits.max_functions,
    )?;
    let selection = semantic.select_kernel_body_v1().ok_or_else(|| {
        unsupported(
            0,
            None,
            None,
            "one direct kernel body or an exact transparent KernelResult wrapper is required",
        )
    })?;
    if !semantic.allocations().is_empty()
        || !semantic.statics().is_empty()
        || !semantic.vtables().is_empty()
    {
        return Err(unsupported(
            0,
            None,
            None,
            "allocations, statics, and vtables are not lowered yet",
        ));
    }
    let root = semantic
        .functions()
        .get(selection.root().index() as usize)
        .ok_or_else(|| unsupported(0, None, None, "the selected kernel root is missing"))?;
    let function = semantic
        .functions()
        .get(selection.body().index() as usize)
        .ok_or_else(|| unsupported(0, None, None, "the selected kernel body is missing"))?;
    let entry = root
        .kernel_entry()
        .ok_or_else(|| unsupported(0, None, None, "kernel export metadata is missing"))?;
    let symbol = std::str::from_utf8(entry.export_symbol().as_bytes())
        .map_err(|_| unsupported(0, None, None, "kernel export symbol is not UTF-8"))?;
    let required_workgroup = entry
        .source_contract()
        .launch()
        .and_then(|launch| launch.required())
        .map(|required| required.as_array());
    let infallible_asserts = match (authenticated_launch_rank, required_workgroup) {
        (Some(_), Some(required_workgroup)) => InfallibleBoundsAssertAnalysisV1::analyze(
            semantic.types(),
            semantic.callables(),
            function,
            required_workgroup,
        )?,
        (None, _) | (_, None) => BTreeSet::new(),
    };
    let has_runtime_assert =
        semantic_requires_runtime_assert_failure(function, &infallible_asserts);
    let lowered_block_count = function
        .blocks()
        .len()
        .checked_add(usize::from(has_runtime_assert))
        .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Blocks,
            actual: usize::MAX,
            limit: limits.max_blocks,
        })?;
    enforce_limit(
        ProductionSemanticKirResourceV1::Blocks,
        lowered_block_count,
        limits.max_blocks,
    )?;
    let statement_count = function
        .blocks()
        .iter()
        .try_fold(0_usize, |count, block| {
            count.checked_add(block.statements().len())
        })
        .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Statements,
            actual: usize::MAX,
            limit: limits.max_statements,
        })?;
    enforce_limit(
        ProductionSemanticKirResourceV1::Statements,
        statement_count,
        limits.max_statements,
    )?;

    let mut parameters = function
        .locals()
        .iter()
        .enumerate()
        .filter_map(|(local, declaration)| match declaration.role() {
            SemanticLocalRoleV1::Argument(argument) => Some((argument, local, declaration.ty())),
            SemanticLocalRoleV1::Return | SemanticLocalRoleV1::Temporary => None,
        })
        .collect::<Vec<_>>();
    parameters.sort_by_key(|(argument, _, _)| *argument);
    if parameters
        .iter()
        .enumerate()
        .any(|(expected, (actual, _, _))| usize::try_from(*actual) != Ok(expected))
    {
        return Err(unsupported(
            0,
            None,
            None,
            "kernel argument locals are not contiguous",
        ));
    }
    let parameter_types = parameters
        .iter()
        .map(|(argument, _, ty)| {
            lower_kernel_parameter_type(
                semantic.types(),
                semantic.callables(),
                function,
                *argument,
                *ty,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameter_values = parameters
        .iter()
        .map(|(_, local, _)| {
            u32::try_from(*local)
                .map(ValueId)
                .map_err(|_| unsupported(0, None, None, "local identity does not fit Kernel IR"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut parameter_bindings = Vec::new();
    parameter_bindings
        .try_reserve_exact(parameters.len())
        .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
            resource: ProductionSemanticKirResourceV1::DebugBindings,
        })?;
    parameter_bindings.extend(parameters.iter().zip(&parameter_values).map(
        |((_, _, _), value)| SemanticKirParameterBindingV1 {
            semantic_function: selection.body(),
            semantic_local: SemanticLocalIdV1::from_index(value.0),
            kernel_ir_value: *value,
        },
    ));
    let mut lowering = SemanticFunctionLoweringV1::new(
        semantic.types(),
        semantic.callables(),
        function,
        SemanticParameterBindingsV1 {
            declarations: &parameters,
            values: &parameter_values,
            types: &parameter_types,
        },
        has_runtime_assert.then(|| BlockId(function.blocks().len() as u32)),
        required_workgroup,
        infallible_asserts,
        launch_rank,
        authenticated_launch_rank.is_some(),
        limits.max_operations,
    )?;

    let order = semantic_cfg_reverse_postorder(function)?;
    let mut blocks = Vec::with_capacity(order.len());
    let mut correspondence = Vec::with_capacity(order.len());
    let mut statement_operation_spans = Vec::with_capacity(statement_count);
    let mut terminator_operation_spans = Vec::with_capacity(order.len());
    let mut synthetic_operation_spans = Vec::with_capacity(usize::from(has_runtime_assert));
    for semantic_block in order {
        let index = usize::try_from(semantic_block.index())
            .map_err(|_| unsupported(0, None, None, "block identity does not fit this host"))?;
        let source = function.blocks().get(index).ok_or_else(|| {
            unsupported(0, Some(semantic_block.index()), None, "block is missing")
        })?;
        let mut target = BasicBlock::new(BlockId(semantic_block.index()));
        let prologue_first = target.operations.len();
        lowering.begin_block(semantic_block, &mut target)?;
        let (first_operation_ordinal, operation_count) =
            measured_operation_span(prologue_first, target.operations.len(), target.id, None)?;
        if operation_count != 0 {
            synthetic_operation_spans.push(SemanticKirSyntheticOperationSpanV1 {
                rule: SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage,
                kernel_ir_block: target.id,
                first_operation_ordinal,
                operation_count,
            });
        }
        for (statement, operation) in source.statements().iter().enumerate() {
            let statement = u32::try_from(statement).map_err(|_| {
                unsupported(
                    0,
                    Some(semantic_block.index()),
                    None,
                    "statement ordinal is too large",
                )
            })?;
            let first = target.operations.len();
            lowering.lower_statement(
                semantic_block,
                Some(statement),
                operation.kind(),
                &mut target.operations,
            )?;
            let (first_operation_ordinal, operation_count) = measured_operation_span(
                first,
                target.operations.len(),
                target.id,
                Some(statement),
            )?;
            statement_operation_spans.push(SemanticKirStatementOperationSpanV1 {
                semantic_function: selection.body(),
                semantic_block,
                statement_ordinal: statement,
                kernel_ir_block: target.id,
                first_operation_ordinal,
                operation_count,
            });
        }
        let terminator_first = target.operations.len();
        target.terminator = Some(lowering.lower_terminator(
            semantic_block,
            source.terminator().kind(),
            &mut target.operations,
        )?);
        let (first_operation_ordinal, operation_count) =
            measured_operation_span(terminator_first, target.operations.len(), target.id, None)?;
        terminator_operation_spans.push(SemanticKirTerminatorOperationSpanV1 {
            semantic_function: selection.body(),
            semantic_block,
            kernel_ir_block: target.id,
            first_operation_ordinal,
            operation_count,
        });
        blocks.push(target);
        correspondence.push(SemanticKirBlockCorrespondenceV1 {
            semantic_function: selection.body(),
            semantic_block,
            kernel_ir_block: BlockId(semantic_block.index()),
            source_statement_count: u32::try_from(source.statements().len()).map_err(|_| {
                unsupported(
                    0,
                    Some(semantic_block.index()),
                    None,
                    "statement count is too large",
                )
            })?,
        });
    }
    if let Some(failure_block) = lowering.assert_failure_block {
        let mut block = BasicBlock::new(failure_block);
        let first = block.operations.len();
        lowering.push_operation(&mut block.operations, || {
            AmdGpuDiagnosticOperation::Trap.operation(None)
        })?;
        let (first_operation_ordinal, operation_count) =
            measured_operation_span(first, block.operations.len(), failure_block, None)?;
        synthetic_operation_spans.push(SemanticKirSyntheticOperationSpanV1 {
            rule: SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap,
            kernel_ir_block: failure_block,
            first_operation_ordinal,
            operation_count,
        });
        block.terminator = Some(Terminator::Unreachable);
        blocks.push(block);
    }
    let operation_capabilities = blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .flat_map(Operation::required_capabilities)
        .collect::<BTreeSet<_>>();
    let diagnostic_declarations = blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|operation| match &operation.kind {
            OperationKind::Call { callee, arguments } => {
                AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
            }
            _ => None,
        })
        .map(|operation| {
            let declaration = operation.declaration();
            (declaration.id.clone(), declaration)
        })
        .collect::<BTreeMap<_, _>>();
    let float_declarations = blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|operation| match &operation.kind {
            OperationKind::Call { callee, arguments } => {
                FloatOperation::from_intrinsic_call(callee, arguments)
            }
            _ => None,
        })
        .map(|operation| {
            let declaration = operation.declaration();
            (declaration.id.clone(), declaration)
        })
        .collect::<BTreeMap<_, _>>();

    let function_id = FunctionId::new(symbol);
    let mut module = Module::new(format!(
        "fe2o3::semantic::{}",
        hex_identity(semantic.semantic_sha256().as_bytes())
    ));
    let trap = AmdGpuDiagnosticOperation::Trap;
    if has_runtime_assert {
        module
            .required_capabilities
            .extend(trap.required_capabilities());
    }
    module
        .required_capabilities
        .extend(operation_capabilities.iter().cloned());
    let mut entry_function = Function::kernel_entry(
        function_id.clone(),
        Signature::new(parameter_types, vec![]),
        parameter_values,
        blocks,
    );
    if has_runtime_assert {
        entry_function
            .required_capabilities
            .extend(trap.required_capabilities());
    }
    entry_function
        .required_capabilities
        .extend(operation_capabilities.iter().cloned());
    module.functions.push(entry_function);
    module
        .functions
        .extend(diagnostic_declarations.into_values());
    module.functions.extend(float_declarations.into_values());
    let required_workgroup = entry
        .source_contract()
        .launch()
        .and_then(|launch| launch.required());
    let dimensions = required_workgroup.map(|required| required.as_array());
    let launch = match (launch_rank, dimensions) {
        (1, Some([_, 1, 1]) | None) => LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
        (2, Some([_, _, 1]) | None) => LaunchDomain::D2 {
            x: LaunchExtent::Dynamic,
            y: LaunchExtent::Dynamic,
        },
        (3, Some(_) | None) => LaunchDomain::D3 {
            x: LaunchExtent::Dynamic,
            y: LaunchExtent::Dynamic,
            z: LaunchExtent::Dynamic,
        },
        _ => {
            return Err(unsupported(
                0,
                None,
                None,
                "authenticated launch rank disagrees with source workgroup axes",
            ));
        }
    };
    let mut kernel = Kernel::new(symbol, function_id, launch);
    if let Some(required) = required_workgroup {
        let [x, y, z] = required.as_array();
        kernel.workgroup_size = Some(WorkgroupSize::new(x, y, z));
    }
    if has_runtime_assert {
        kernel
            .required_capabilities
            .extend(trap.required_capabilities());
    }
    kernel.required_capabilities.extend(operation_capabilities);
    module.kernels.push(kernel);

    let correspondence = SemanticKirCorrespondenceV1 {
        semantic_sha256: *semantic.semantic_sha256().as_bytes(),
        function_count: semantic.functions().len(),
        blocks: correspondence.into_boxed_slice(),
        statement_operation_spans: statement_operation_spans.into_boxed_slice(),
        terminator_operation_spans: terminator_operation_spans.into_boxed_slice(),
        synthetic_operation_spans: synthetic_operation_spans.into_boxed_slice(),
        parameter_bindings: parameter_bindings.into_boxed_slice(),
    };
    correspondence.validate_layout_against(owner, &module, has_runtime_assert)?;
    Ok((module, correspondence))
}

fn semantic_cfg_reverse_postorder(
    function: &SemanticFunctionDeclV1,
) -> Result<Vec<SemanticBlockIdV1>, ProductionSemanticKirErrorV1> {
    let mut postorder = Vec::with_capacity(function.blocks().len());
    let mut visited = vec![false; function.blocks().len()];
    let mut stack = vec![(function.entry(), false)];
    while let Some((block, expanded)) = stack.pop() {
        let index = usize::try_from(block.index())
            .map_err(|_| unsupported(0, None, None, "block identity does not fit this host"))?;
        let Some(source) = function.blocks().get(index) else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "CFG traversal references a missing block",
            ));
        };
        if expanded {
            postorder.push(block);
            continue;
        }
        if visited[index] {
            continue;
        }
        visited[index] = true;
        stack.push((block, true));
        let mut successors = Vec::with_capacity(source.terminator().kind().edge_count());
        source
            .terminator()
            .kind()
            .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                let target = edge.target();
                let target_index = usize::try_from(target.index()).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "CFG successor identity does not fit this host",
                    )
                })?;
                if target_index >= function.blocks().len() {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "CFG traversal references a missing successor",
                    ));
                }
                successors.push(target);
                Ok(())
            })?;
        stack.extend(
            successors
                .into_iter()
                .rev()
                .map(|successor| (successor, false)),
        );
    }
    postorder.reverse();
    let mut order = postorder;
    order.extend(
        visited
            .iter()
            .enumerate()
            .filter(|(_, seen)| !**seen)
            .map(|(index, _)| SemanticBlockIdV1::from_index(index as u32)),
    );
    Ok(order)
}

const MAX_PROMOTED_LOCALS_V1: usize = 128;
const MAX_PROMOTED_BLOCK_PARAMETERS_V1: usize = 16_384;

#[derive(Clone, Debug)]
struct SemanticPromotedLocalV1 {
    semantic_type: SemanticTypeIdV1,
    binding: SemanticPromotedBindingV1,
    kernel_types: Box<[Type]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticPromotedBindingV1 {
    Ordinary,
    WorkgroupCollectiveScratch {
        element: SemanticTypeIdV1,
    },
    MatrixFragment {
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
    },
    AccumulatorFragment {
        contract: SemanticMfmaAccumulatorContractV1,
    },
    Gfx950LdsTransposeTile {
        format: SemanticGfx950LdsTransposeFormatV1,
        state: SemanticGfx950LdsTransposeStateV1,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticGfx950LdsTransposeStateV1 {
    Uninitialized,
    Staged,
    Published,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticCurrentWaveV1 {
    width: u32,
}

impl SemanticCurrentWaveV1 {
    const fn new(width: u32) -> Self {
        Self { width }
    }
}

impl SemanticPromotedBindingV1 {
    fn transport_types(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
    ) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
        let transport = match self {
            Self::Ordinary => lower_ssa_value_types(types, semantic_type)?,
            Self::WorkgroupCollectiveScratch { element } => {
                lower_workgroup_collective_scratch_transport_v1(types, semantic_type, element)?
            }
            Self::MatrixFragment { contract, .. } => match contract.profile {
                SemanticMfmaProfileV1::Bf16F32M16N16K16 => {
                    vec![Type::Scalar(ScalarType::Bf16); 4]
                }
                SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128 => {
                    vec![Type::Scalar(ScalarType::U32); 8]
                }
            },
            Self::AccumulatorFragment { contract } => match contract.profile {
                SemanticMfmaProfileV1::Bf16F32M16N16K16
                | SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128 => {
                    vec![Type::Scalar(ScalarType::F32); 4]
                }
            },
            Self::Gfx950LdsTransposeTile { .. } => vec![gfx950_lds_transpose_pointer_type_v1()],
        };
        Ok(transport)
    }

    const fn current_wave(self) -> Option<SemanticCurrentWaveV1> {
        match self {
            Self::Ordinary | Self::WorkgroupCollectiveScratch { .. } => None,
            Self::MatrixFragment { contract, .. } => {
                Some(SemanticCurrentWaveV1::new(contract.wave_width))
            }
            Self::AccumulatorFragment { contract } => {
                Some(SemanticCurrentWaveV1::new(contract.wave_width))
            }
            Self::Gfx950LdsTransposeTile { .. } => Some(SemanticCurrentWaveV1::new(64)),
        }
    }

    fn transport_values(
        self,
        binding: &SemanticValueBindingV1,
    ) -> Result<Vec<(ValueId, Type)>, &'static str> {
        match (self, binding) {
            (Self::Ordinary, binding) => binding.values(),
            (Self::WorkgroupCollectiveScratch { .. }, SemanticValueBindingV1::Aggregate(_)) => {
                binding.values()
            }
            (
                Self::MatrixFragment {
                    contract,
                    storage_layout,
                },
                SemanticValueBindingV1::MatrixFragment {
                    values,
                    contract: actual_contract,
                    storage_layout: actual_storage_layout,
                    wave,
                },
            ) if contract == *actual_contract
                && storage_layout == *actual_storage_layout
                && self.current_wave() == Some(*wave) =>
            {
                Ok(values.clone())
            }
            (
                Self::AccumulatorFragment { contract },
                SemanticValueBindingV1::AccumulatorFragment {
                    values,
                    contract: actual_contract,
                    wave,
                },
            ) if contract == *actual_contract && self.current_wave() == Some(*wave) => {
                Ok(values.clone())
            }
            (
                Self::Gfx950LdsTransposeTile { format, state },
                SemanticValueBindingV1::Gfx950LdsTransposeTile {
                    storage,
                    format: actual_format,
                    state: actual_state,
                },
            ) if format == *actual_format && state == *actual_state => {
                Ok(vec![(*storage, gfx950_lds_transpose_pointer_type_v1())])
            }
            (Self::MatrixFragment { .. }, _) => {
                Err("promoted matrix fragment lacks its authenticated producer metadata")
            }
            (Self::AccumulatorFragment { .. }, _) => {
                Err("promoted accumulator fragment lacks its authenticated producer metadata")
            }
            (Self::Gfx950LdsTransposeTile { .. }, _) => {
                Err("promoted gfx950 LDS transpose tile lacks its authenticated state")
            }
            (Self::WorkgroupCollectiveScratch { .. }, _) => {
                Err("promoted workgroup scratch lacks its authenticated aggregate")
            }
        }
    }

    fn binding_from_transport(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        values: &[ValueDef],
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        if matches!(self, Self::Ordinary) {
            return binding_from_value_defs(types, semantic_type, values);
        }
        let expected = self.transport_types(types, semantic_type)?;
        if values.len() != expected.len()
            || values
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| &actual.ty != expected)
        {
            return Err(unsupported(
                0,
                None,
                None,
                "typed fragment SSA component types changed",
            ));
        }
        let components = values
            .iter()
            .map(|value| (value.id, value.ty.clone()))
            .collect();
        match self {
            Self::Ordinary => unreachable!("ordinary binding returned above"),
            Self::WorkgroupCollectiveScratch { .. } => {
                binding_from_value_defs_with_validation(types, semantic_type, values, false)
            }
            Self::MatrixFragment {
                contract,
                storage_layout,
            } => Ok(SemanticValueBindingV1::MatrixFragment {
                values: components,
                contract,
                storage_layout,
                wave: self
                    .current_wave()
                    .expect("matrix fragments have a current-wave association"),
            }),
            Self::AccumulatorFragment { contract } => {
                Ok(SemanticValueBindingV1::AccumulatorFragment {
                    values: components,
                    contract,
                    wave: self
                        .current_wave()
                        .expect("accumulator fragments have a current-wave association"),
                })
            }
            Self::Gfx950LdsTransposeTile { format, state } => {
                Ok(SemanticValueBindingV1::Gfx950LdsTransposeTile {
                    storage: components[0].0,
                    format,
                    state,
                })
            }
        }
    }
}

fn gfx950_lds_transpose_pointer_type_v1() -> Type {
    Type::pointer(
        Type::Scalar(ScalarType::U8),
        AddressSpace::Workgroup,
        AccessMode::ReadWrite,
    )
}

fn insert_compiler_issued_ssa_binding_v1(
    bindings: &mut BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    ty: SemanticTypeIdV1,
    binding: SemanticPromotedBindingV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if let Some(existing) = bindings.insert(ty, binding)
        && existing != binding
    {
        return Err(unsupported(
            0,
            None,
            None,
            "one semantic fragment type has conflicting compiler-issued contracts",
        ));
    }
    Ok(())
}

fn compiler_issued_ssa_bindings_v1(
    callables: &[SemanticCallableDeclV1],
) -> Result<BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>, ProductionSemanticKirErrorV1> {
    let mut bindings = BTreeMap::new();
    for callable in callables {
        let SemanticCallableDeclV1::CompilerIntrinsic { operation, .. } = callable else {
            continue;
        };
        require_current_production_intrinsic_v1(operation)?;
        match operation {
            SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoadZeroFilledV2 {
                fragment,
                contract,
                storage_layout,
                ..
            }
            | SemanticCompilerIntrinsicOperationV1::Gfx950Fp4MatrixLoadM16K128 {
                fragment,
                contract,
                storage_layout,
                ..
            }
            | SemanticCompilerIntrinsicOperationV1::Gfx950Fp8MatrixLoadM16K128 {
                fragment,
                contract,
                storage_layout,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *fragment,
                SemanticPromotedBindingV1::MatrixFragment {
                    contract: *contract,
                    storage_layout: *storage_layout,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeCurrent {
                tile,
                format,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *tile,
                SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                    format: *format,
                    state: SemanticGfx950LdsTransposeStateV1::Uninitialized,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeStage {
                output_tile,
                format,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *output_tile,
                SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                    format: *format,
                    state: SemanticGfx950LdsTransposeStateV1::Staged,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposePublish {
                output_tile,
                format,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *output_tile,
                SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                    format: *format,
                    state: SemanticGfx950LdsTransposeStateV1::Published,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeRead {
                fragment,
                contract,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *fragment,
                SemanticPromotedBindingV1::MatrixFragment {
                    contract: *contract,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorZero {
                fragment,
                contract,
                ..
            }
            | SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                accumulator_fragment: fragment,
                accumulator: contract,
                ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *fragment,
                SemanticPromotedBindingV1::AccumulatorFragment {
                    contract: *contract,
                },
            )?,
            SemanticCompilerIntrinsicOperationV1::WorkgroupReduceSum {
                scratch, element, ..
            } => insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                *scratch,
                SemanticPromotedBindingV1::WorkgroupCollectiveScratch { element: *element },
            )?,
            _ => {}
        }
    }

    for callable in callables {
        let SemanticCallableDeclV1::CompilerIntrinsic { operation, .. } = callable else {
            continue;
        };
        match operation {
            SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                lhs_fragment,
                rhs_fragment,
                lhs,
                rhs,
                ..
            } => {
                for (fragment, expected) in [(*lhs_fragment, *lhs), (*rhs_fragment, *rhs)] {
                    if let Some(binding) = bindings.get(&fragment)
                        && !matches!(
                            binding,
                            SemanticPromotedBindingV1::MatrixFragment { contract, .. }
                                if *contract == expected
                        )
                    {
                        return Err(unsupported(
                            0,
                            None,
                            None,
                            "matrix consumer contract conflicts with its compiler-issued fragment type",
                        ));
                    }
                }
            }
            SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorIntoValues {
                fragment,
                ..
            } => {
                if let Some(binding) = bindings.get(fragment)
                    && !matches!(
                        binding,
                        SemanticPromotedBindingV1::AccumulatorFragment { .. }
                    )
                {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "accumulator projection conflicts with its compiler-issued fragment type",
                    ));
                }
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeStage {
                input_tile,
                format,
                ..
            } => {
                if let Some(binding) = bindings.get(input_tile)
                    && !matches!(
                        binding,
                        SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                            format: actual_format,
                            state: SemanticGfx950LdsTransposeStateV1::Uninitialized,
                        } if actual_format == format
                    )
                {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "gfx950 LDS transpose stage input has conflicting compiler-issued state",
                    ));
                }
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposePublish {
                input_tile,
                format,
                ..
            } => {
                if let Some(binding) = bindings.get(input_tile)
                    && !matches!(
                        binding,
                        SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                            format: actual_format,
                            state: SemanticGfx950LdsTransposeStateV1::Staged,
                        } if actual_format == format
                    )
                {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "gfx950 LDS transpose publish input has conflicting compiler-issued state",
                    ));
                }
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeRead {
                tile, format, ..
            } => {
                if let Some(binding) = bindings.get(tile)
                    && !matches!(
                        binding,
                        SemanticPromotedBindingV1::Gfx950LdsTransposeTile {
                            format: actual_format,
                            state: SemanticGfx950LdsTransposeStateV1::Published,
                        } if actual_format == format
                    )
                {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "gfx950 LDS transpose read input has conflicting compiler-issued state",
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(bindings)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SemanticWorkgroupPipelineTypeContractV1 {
    element: SemanticTypeIdV1,
    payload_binding: SemanticPromotedBindingV1,
    component_types: Box<[Type]>,
    packed_type: Type,
    alignment: u32,
}

fn pipeline_scalar_bit_width_v1(ty: &Type) -> Option<u32> {
    match ty.as_scalar()? {
        ScalarType::Index => Some(64),
        ScalarType::Bool => Some(8),
        scalar => scalar.bit_width().map(u32::from),
    }
}

fn pipeline_packed_type_v1(bits: u32) -> Option<Type> {
    Some(Type::Scalar(match bits {
        8 => ScalarType::U8,
        16 => ScalarType::U16,
        32 => ScalarType::U32,
        64 => ScalarType::U64,
        128 => ScalarType::U128,
        _ => return None,
    }))
}

fn pipeline_unsigned_component_type_v1(ty: &Type) -> Option<Type> {
    Some(Type::Scalar(match pipeline_scalar_bit_width_v1(ty)? {
        8 => ScalarType::U8,
        16 => ScalarType::U16,
        32 => ScalarType::U32,
        64 => ScalarType::U64,
        128 => ScalarType::U128,
        _ => return None,
    }))
}

fn pipeline_integer_constant_v1(ty: &Type, value: u64) -> Option<Constant> {
    match ty.as_scalar()? {
        ScalarType::U8 => u8::try_from(value).ok().map(Constant::U8),
        ScalarType::U16 => u16::try_from(value).ok().map(Constant::U16),
        ScalarType::U32 => u32::try_from(value).ok().map(Constant::U32),
        ScalarType::U64 => Some(Constant::U64(value)),
        _ => None,
    }
}

fn workgroup_pipeline_type_contracts_v1(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    compiler_issued_bindings: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
) -> Result<
    BTreeMap<SemanticTypeIdV1, SemanticWorkgroupPipelineTypeContractV1>,
    ProductionSemanticKirErrorV1,
> {
    let mut elements = BTreeMap::new();
    for callable in callables {
        let SemanticCallableDeclV1::CompilerIntrinsic { operation, .. } = callable else {
            continue;
        };
        let (pipeline, element) = match operation {
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite { pipeline, element }
            | SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineRead { pipeline, element } => {
                (*pipeline, *element)
            }
            _ => continue,
        };
        if let Some(existing) = elements.insert(pipeline, element)
            && existing != element
        {
            return Err(unsupported(
                0,
                None,
                None,
                "one workgroup pipeline type has inconsistent payload types",
            ));
        }
    }

    let mut contracts = BTreeMap::new();
    for (pipeline, element) in elements {
        let payload_binding = compiler_issued_bindings
            .get(&element)
            .copied()
            .unwrap_or(SemanticPromotedBindingV1::Ordinary);
        let component_types = payload_binding.transport_types(types, element)?;
        if component_types.is_empty() {
            return Err(unsupported(
                0,
                None,
                None,
                "workgroup pipeline payload has no physical components",
            ));
        }
        let component_bits = component_types.iter().try_fold(0_u32, |total, component| {
            let bits = pipeline_scalar_bit_width_v1(component).ok_or_else(|| {
                unsupported(
                    0,
                    None,
                    None,
                    "workgroup pipeline payload component is not a physical scalar",
                )
            })?;
            total.checked_add(bits).ok_or_else(|| {
                unsupported(0, None, None, "workgroup pipeline payload width overflows")
            })
        })?;
        let declaration = types.get(element.index() as usize).ok_or_else(|| {
            unsupported(0, None, None, "workgroup pipeline payload type is missing")
        })?;
        let layout_bits = declaration
            .layout()
            .size_bytes()
            .and_then(|bytes| bytes.checked_mul(8))
            .and_then(|bits| u32::try_from(bits).ok())
            .ok_or_else(|| {
                unsupported(
                    0,
                    None,
                    None,
                    "workgroup pipeline payload layout width is unavailable",
                )
            })?;
        if component_bits != layout_bits {
            return Err(unsupported(
                0,
                None,
                None,
                "workgroup pipeline payload transport does not cover its exact Rust layout",
            ));
        }
        let packed_type = pipeline_packed_type_v1(layout_bits).ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "workgroup pipeline payload has no exact packed Kernel IR scalar",
            )
        })?;
        if layout_bits == 128 && component_types.as_slice() != [packed_type.clone()] {
            return Err(unsupported(
                0,
                None,
                None,
                "composite 128-bit workgroup pipeline packing is not executable",
            ));
        }
        let source_alignment = u32::try_from(declaration.layout().alignment_bytes())
            .ok()
            .filter(|alignment| *alignment != 0)
            .ok_or_else(|| {
                unsupported(
                    0,
                    None,
                    None,
                    "workgroup pipeline payload alignment is unavailable",
                )
            })?;
        let packed_alignment = layout_bits
            .checked_div(8)
            .filter(|alignment| *alignment != 0)
            .ok_or_else(|| {
                unsupported(
                    0,
                    None,
                    None,
                    "workgroup pipeline packed alignment is unavailable",
                )
            })?;
        let alignment = source_alignment.max(packed_alignment);
        contracts.insert(
            pipeline,
            SemanticWorkgroupPipelineTypeContractV1 {
                element,
                payload_binding,
                component_types: component_types.into_boxed_slice(),
                packed_type,
                alignment,
            },
        );
    }
    Ok(contracts)
}

fn require_current_production_intrinsic_v1(
    operation: &SemanticCompilerIntrinsicOperationV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if matches!(
        operation,
        SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoad { .. }
    ) {
        Err(unsupported(
            0,
            None,
            None,
            "the retired Option-returning BF16 matrix load is not admitted; use Bf16MatrixLoadZeroFilledV2",
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct SemanticControlFlowSsaPlanV1 {
    compiler_issued_bindings: BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    promoted: BTreeMap<u32, SemanticPromotedLocalV1>,
    live_in: BTreeMap<u32, Vec<u32>>,
}

impl SemanticControlFlowSsaPlanV1 {
    fn analyze(
        types: &[SemanticTypeDeclV1],
        callables: &[SemanticCallableDeclV1],
        function: &SemanticFunctionDeclV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let compiler_issued_bindings = compiler_issued_ssa_bindings_v1(callables)?;
        let mut definition_counts = vec![0_u32; function.locals().len()];
        let mut projected = BTreeSet::new();
        for block in function.blocks() {
            for statement in block.statements() {
                if let SemanticStatementKindV1::Assign(assignment) = statement.kind() {
                    let destination = assignment.destination();
                    if destination.projections().is_empty() {
                        let count = definition_counts
                            .get_mut(destination.local().index() as usize)
                            .ok_or_else(|| {
                                unsupported(0, None, None, "assignment local is missing")
                            })?;
                        *count = count.saturating_add(1);
                    } else {
                        projected.insert(destination.local().index());
                    }
                }
            }
            if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
                && let Some(destination) = call.destination()
            {
                if destination.place().projections().is_empty() {
                    let count = definition_counts
                        .get_mut(destination.place().local().index() as usize)
                        .ok_or_else(|| {
                            unsupported(0, None, None, "call destination local is missing")
                        })?;
                    *count = count.saturating_add(1);
                } else {
                    projected.insert(destination.place().local().index());
                }
            }
        }

        let mut promoted = BTreeMap::new();
        for (local, declaration) in function.locals().iter().enumerate() {
            if definition_counts[local] < 2 || projected.contains(&(local as u32)) {
                continue;
            }
            let binding = compiler_issued_bindings
                .get(&declaration.ty())
                .copied()
                .unwrap_or(SemanticPromotedBindingV1::Ordinary);
            if let Ok(kernel_types) = binding.transport_types(types, declaration.ty())
                && !kernel_types.is_empty()
            {
                promoted.insert(
                    local as u32,
                    SemanticPromotedLocalV1 {
                        semantic_type: declaration.ty(),
                        binding,
                        kernel_types: kernel_types.into_boxed_slice(),
                    },
                );
            }
        }
        if promoted.is_empty() {
            return Ok(Self {
                compiler_issued_bindings,
                promoted,
                live_in: BTreeMap::new(),
            });
        }
        if promoted.len() > MAX_PROMOTED_LOCALS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "mutable control flow exceeds the promoted-local limit",
            ));
        }

        let block_ids = (0..function.blocks().len())
            .map(|block| block as u32)
            .collect::<BTreeSet<_>>();
        let mut uses = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut defs = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut successors = BTreeMap::<u32, Vec<u32>>::new();
        for (block_index, block) in function.blocks().iter().enumerate() {
            let block_id = block_index as u32;
            let mut block_uses = BTreeSet::new();
            let mut block_defs = BTreeSet::new();
            for statement in block.statements() {
                collect_statement_uses_v1(
                    statement.kind(),
                    &promoted,
                    &block_defs,
                    &mut block_uses,
                );
                if let SemanticStatementKindV1::Assign(assignment) = statement.kind()
                    && assignment.destination().projections().is_empty()
                    && promoted.contains_key(&assignment.destination().local().index())
                {
                    block_defs.insert(assignment.destination().local().index());
                }
            }
            collect_terminator_uses_v1(
                block.terminator().kind(),
                &promoted,
                &block_defs,
                &mut block_uses,
            );
            if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
                && let Some(destination) = call.destination()
                && destination.place().projections().is_empty()
                && promoted.contains_key(&destination.place().local().index())
            {
                block_defs.insert(destination.place().local().index());
            }
            let mut block_successors = Vec::new();
            block
                .terminator()
                .kind()
                .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                    if !block_ids.contains(&edge.target().index()) {
                        return Err(unsupported(
                            0,
                            Some(block_id),
                            None,
                            "CFG successor is missing",
                        ));
                    }
                    block_successors.push(edge.target().index());
                    Ok(())
                })?;
            uses.insert(block_id, block_uses);
            defs.insert(block_id, block_defs);
            successors.insert(block_id, block_successors);
        }

        let mut live_in = function
            .blocks()
            .iter()
            .enumerate()
            .map(|(block, _)| (block as u32, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let mut predecessors = function
            .blocks()
            .iter()
            .enumerate()
            .map(|(block, _)| (block as u32, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        for (source, targets) in &successors {
            for target in targets {
                predecessors
                    .get_mut(target)
                    .expect("validated successor")
                    .insert(*source);
            }
        }
        let mut worklist = (0..function.blocks().len() as u32).collect::<VecDeque<_>>();
        let mut queued = (0..function.blocks().len() as u32).collect::<BTreeSet<_>>();
        while let Some(block_id) = worklist.pop_front() {
            queued.remove(&block_id);
            let live_out = successors[&block_id]
                .iter()
                .flat_map(|target| live_in[target].iter().copied())
                .collect::<BTreeSet<_>>();
            let mut next = uses[&block_id].clone();
            next.extend(live_out.difference(&defs[&block_id]).copied());
            if next != live_in[&block_id] {
                live_in.insert(block_id, next);
                for predecessor in &predecessors[&block_id] {
                    if queued.insert(*predecessor) {
                        worklist.push_back(*predecessor);
                    }
                }
            }
        }

        let entry = function.entry().index();
        if live_in[&entry].iter().any(|local| {
            !matches!(
                function.locals()[*local as usize].role(),
                SemanticLocalRoleV1::Argument(_)
            )
        }) {
            return Err(unsupported(
                0,
                Some(entry),
                None,
                "mutable scalar control flow reads a local before its entry definition",
            ));
        }
        let parameter_count = live_in
            .iter()
            .filter(|(block, _)| **block != entry)
            .flat_map(|(_, locals)| locals)
            .map(|local| promoted[local].kernel_types.len())
            .sum::<usize>();
        if parameter_count > MAX_PROMOTED_BLOCK_PARAMETERS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "mutable scalar control flow exceeds the block-parameter limit",
            ));
        }
        for (block, targets) in &successors {
            let mut seen = BTreeSet::new();
            for target in targets {
                if !seen.insert(*target) && !live_in[target].is_empty() {
                    return Err(unsupported(
                        0,
                        Some(*block),
                        None,
                        "multiple live-value edges from one block to one successor are unsupported",
                    ));
                }
            }
        }

        Ok(Self {
            compiler_issued_bindings,
            promoted,
            live_in: live_in
                .into_iter()
                .map(|(block, locals)| (block, locals.into_iter().collect()))
                .collect(),
        })
    }

    fn live_in(&self, block: u32) -> &[u32] {
        self.live_in.get(&block).map_or(&[], Vec::as_slice)
    }
}

fn analyze_promoted_enum_variants_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    control_flow_ssa: &SemanticControlFlowSsaPlanV1,
) -> Result<BTreeMap<(u32, u32), u32>, ProductionSemanticKirErrorV1> {
    type ExactVariantsV1 = BTreeMap<u32, u32>;

    fn whole_local(operand: &SemanticOperandV1) -> Option<u32> {
        match operand {
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)
                if place.projections().is_empty() =>
            {
                Some(place.local().index())
            }
            SemanticOperandV1::Copy(_)
            | SemanticOperandV1::Move(_)
            | SemanticOperandV1::Constant(_) => None,
        }
    }

    fn meet(existing: &ExactVariantsV1, incoming: &ExactVariantsV1) -> ExactVariantsV1 {
        existing
            .iter()
            .filter_map(|(local, variant)| {
                (incoming.get(local) == Some(variant)).then_some((*local, *variant))
            })
            .collect()
    }

    let promoted_enums = control_flow_ssa
        .promoted
        .iter()
        .filter_map(|(local, promoted)| {
            types
                .get(promoted.semantic_type.index() as usize)
                .is_some_and(|declaration| {
                    matches!(declaration.shape(), SemanticTypeShapeV1::Enum { .. })
                })
                .then_some(*local)
        })
        .collect::<BTreeSet<_>>();
    if promoted_enums.is_empty() {
        return Ok(BTreeMap::new());
    }

    let block_count = function.blocks().len();
    let entry = function.entry().index() as usize;
    let mut incoming = vec![None::<ExactVariantsV1>; block_count];
    let mut queued = BTreeSet::from([function.entry().index()]);
    let mut worklist = VecDeque::from([function.entry().index()]);
    incoming[entry] = Some(BTreeMap::new());

    while let Some(block_index) = worklist.pop_front() {
        queued.remove(&block_index);
        let Some(block) = function.blocks().get(block_index as usize) else {
            return Err(unsupported(
                0,
                Some(block_index),
                None,
                "promoted enum analysis references a missing block",
            ));
        };
        let Some(mut facts) = incoming[block_index as usize].clone() else {
            continue;
        };
        let mut discriminants = BTreeMap::<u32, u32>::new();
        for statement in block.statements() {
            let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                continue;
            };
            let destination = assignment.destination();
            if !destination.projections().is_empty() {
                continue;
            }
            let destination = destination.local().index();
            discriminants.remove(&destination);
            discriminants.retain(|_, source| *source != destination);
            if promoted_enums.contains(&destination) {
                match assignment.value().kind() {
                    SemanticRvalueKindV1::Aggregate(aggregate) => match aggregate.kind() {
                        SemanticAggregateKindV1::EnumVariant(variant) => {
                            facts.insert(destination, *variant);
                        }
                        _ => {
                            facts.remove(&destination);
                        }
                    },
                    _ => {
                        facts.remove(&destination);
                    }
                }
            }
            if let SemanticRvalueKindV1::Discriminant(place) = assignment.value().kind()
                && place.projections().is_empty()
                && promoted_enums.contains(&place.local().index())
            {
                discriminants.insert(destination, place.local().index());
            }
        }

        if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
            && let Some(destination) = call.destination()
            && destination.place().projections().is_empty()
            && promoted_enums.contains(&destination.place().local().index())
        {
            facts.remove(&destination.place().local().index());
        }

        let mut outgoing = Vec::new();
        if let SemanticTerminatorKindV1::SwitchInt {
            discriminant,
            targets,
        } = block.terminator().kind()
            && let Some(enum_local) =
                whole_local(discriminant).and_then(|local| discriminants.get(&local).copied())
        {
            let promoted = &control_flow_ssa.promoted[&enum_local];
            let declaration = types
                .get(promoted.semantic_type.index() as usize)
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block_index),
                        None,
                        "promoted enum switch type is missing",
                    )
                })?;
            let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
                return Err(unsupported(
                    0,
                    Some(block_index),
                    None,
                    "promoted enum switch source is not an enum",
                ));
            };
            for target in targets.values() {
                let mut edge_facts = facts.clone();
                if let Some((variant, _)) = variants
                    .iter()
                    .enumerate()
                    .find(|(_, variant)| variant.discriminant() == target.value())
                {
                    edge_facts.insert(enum_local, variant as u32);
                } else {
                    edge_facts.remove(&enum_local);
                }
                outgoing.push((target.edge().target().index(), edge_facts));
            }
            let mut otherwise = variants.iter().enumerate().filter(|(_, variant)| {
                !variant.is_uninhabited()
                    && !targets
                        .values()
                        .iter()
                        .any(|target| target.value() == variant.discriminant())
            });
            let exact_otherwise = otherwise.next().map(|(variant, _)| variant as u32);
            if otherwise.next().is_some() {
                facts.remove(&enum_local);
            } else if let Some(variant) = exact_otherwise {
                facts.insert(enum_local, variant);
            } else {
                facts.remove(&enum_local);
            }
            outgoing.push((targets.otherwise().target().index(), facts));
        } else {
            block
                .terminator()
                .kind()
                .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                    outgoing.push((edge.target().index(), facts.clone()));
                    Ok(())
                })?;
        }

        for (target, edge_facts) in outgoing {
            let Some(target_incoming) = incoming.get_mut(target as usize) else {
                return Err(unsupported(
                    0,
                    Some(block_index),
                    None,
                    "promoted enum analysis references a missing successor",
                ));
            };
            let next = target_incoming
                .as_ref()
                .map_or_else(|| edge_facts.clone(), |current| meet(current, &edge_facts));
            if target_incoming.as_ref() != Some(&next) {
                *target_incoming = Some(next);
                if queued.insert(target) {
                    worklist.push_back(target);
                }
            }
        }
    }

    Ok(incoming
        .into_iter()
        .enumerate()
        .flat_map(|(block, facts)| {
            facts.into_iter().flat_map(move |facts| {
                facts
                    .into_iter()
                    .map(move |(local, variant)| ((block as u32, local), variant))
            })
        })
        .collect())
}

fn collect_statement_uses_v1(
    statement: &SemanticStatementKindV1,
    promoted: &BTreeMap<u32, SemanticPromotedLocalV1>,
    defs: &BTreeSet<u32>,
    uses: &mut BTreeSet<u32>,
) {
    match statement {
        SemanticStatementKindV1::Assign(assignment) => {
            collect_rvalue_uses_v1(assignment.value().kind(), promoted, defs, uses);
            if !assignment.destination().projections().is_empty() {
                collect_place_use_v1(assignment.destination(), promoted, defs, uses);
            }
        }
        SemanticStatementKindV1::Store(store) => {
            collect_place_use_v1(store.destination(), promoted, defs, uses);
            collect_operand_use_v1(store.value(), promoted, defs, uses);
        }
        _ => {}
    }
}

fn collect_rvalue_uses_v1(
    value: &SemanticRvalueKindV1,
    promoted: &BTreeMap<u32, SemanticPromotedLocalV1>,
    defs: &BTreeSet<u32>,
    uses: &mut BTreeSet<u32>,
) {
    match value {
        SemanticRvalueKindV1::Use(operand)
        | SemanticRvalueKindV1::Unary { operand, .. }
        | SemanticRvalueKindV1::Cast { operand, .. } => {
            collect_operand_use_v1(operand, promoted, defs, uses);
        }
        SemanticRvalueKindV1::Binary { left, right, .. } => {
            collect_operand_use_v1(left, promoted, defs, uses);
            collect_operand_use_v1(right, promoted, defs, uses);
        }
        SemanticRvalueKindV1::CheckedBinary(checked) => {
            collect_operand_use_v1(checked.left(), promoted, defs, uses);
            collect_operand_use_v1(checked.right(), promoted, defs, uses);
        }
        SemanticRvalueKindV1::UncheckedBinary(unchecked) => {
            collect_operand_use_v1(unchecked.left(), promoted, defs, uses);
            collect_operand_use_v1(unchecked.right(), promoted, defs, uses);
        }
        SemanticRvalueKindV1::Borrow { place, .. }
        | SemanticRvalueKindV1::AddressOf { place, .. }
        | SemanticRvalueKindV1::Length(place)
        | SemanticRvalueKindV1::Discriminant(place) => {
            collect_place_use_v1(place, promoted, defs, uses);
        }
        SemanticRvalueKindV1::Aggregate(aggregate) => {
            for operand in aggregate.operands() {
                collect_operand_use_v1(operand, promoted, defs, uses);
            }
        }
        SemanticRvalueKindV1::Load(load) => {
            collect_place_use_v1(load.source(), promoted, defs, uses);
        }
    }
}

fn collect_terminator_uses_v1(
    terminator: &SemanticTerminatorKindV1,
    promoted: &BTreeMap<u32, SemanticPromotedLocalV1>,
    defs: &BTreeSet<u32>,
    uses: &mut BTreeSet<u32>,
) {
    match terminator {
        SemanticTerminatorKindV1::SwitchInt { discriminant, .. } => {
            collect_operand_use_v1(discriminant, promoted, defs, uses);
        }
        SemanticTerminatorKindV1::Call(call) => {
            for argument in call.arguments() {
                collect_operand_use_v1(argument, promoted, defs, uses);
            }
        }
        SemanticTerminatorKindV1::Assert {
            condition, message, ..
        } => {
            collect_operand_use_v1(condition, promoted, defs, uses);
            match message {
                SemanticAssertMessageV1::BoundsCheck { length, index }
                | SemanticAssertMessageV1::Overflow {
                    left: length,
                    right: index,
                    ..
                } => {
                    collect_operand_use_v1(length, promoted, defs, uses);
                    collect_operand_use_v1(index, promoted, defs, uses);
                }
                SemanticAssertMessageV1::DivisionByZero(operand)
                | SemanticAssertMessageV1::RemainderByZero(operand) => {
                    collect_operand_use_v1(operand, promoted, defs, uses);
                }
                SemanticAssertMessageV1::MisalignedPointerDereference {
                    required_alignment,
                    found_alignment,
                } => {
                    collect_operand_use_v1(required_alignment, promoted, defs, uses);
                    collect_operand_use_v1(found_alignment, promoted, defs, uses);
                }
                SemanticAssertMessageV1::NullPointerDereference
                | SemanticAssertMessageV1::ResumedAfterReturn
                | SemanticAssertMessageV1::ResumedAfterPanic => {}
            }
        }
        _ => {}
    }
}

fn collect_operand_use_v1(
    operand: &SemanticOperandV1,
    promoted: &BTreeMap<u32, SemanticPromotedLocalV1>,
    defs: &BTreeSet<u32>,
    uses: &mut BTreeSet<u32>,
) {
    if let SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) = operand {
        collect_place_use_v1(place, promoted, defs, uses);
    }
}

fn collect_place_use_v1(
    place: &SemanticPlaceV1,
    promoted: &BTreeMap<u32, SemanticPromotedLocalV1>,
    defs: &BTreeSet<u32>,
    uses: &mut BTreeSet<u32>,
) {
    let local = place.local().index();
    if promoted.contains_key(&local) && !defs.contains(&local) {
        uses.insert(local);
    }
    for projection in place.projections() {
        if let SemanticProjectionKindV1::Index(index) = projection.kind() {
            let index = index.index();
            if promoted.contains_key(&index) && !defs.contains(&index) {
                uses.insert(index);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadComponentStorageV1 {
    pointer: ValueId,
    kernel_type: Type,
    alignment: u32,
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadFieldStorageV1 {
    semantic_type: SemanticTypeIdV1,
    exact_enum_variant: Option<u32>,
    compiler_issued_binding: Option<SemanticPromotedBindingV1>,
    components: Box<[SemanticEnumPayloadComponentStorageV1]>,
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadSourceV1 {
    place: SemanticPlaceV1,
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadCustodyV1 {
    source_block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
}

#[derive(Clone, Debug)]
enum SemanticEnumPayloadRestoreV1 {
    PrivateStorage(SemanticEnumPayloadFieldStorageV1),
    UniqueSource(SemanticValueBindingV1),
}

#[derive(Clone, Copy, Debug)]
enum SemanticCapabilityAvailabilityV1 {
    Option(SemanticOptionAvailabilityV1),
    EnumPayload {
        local: SemanticLocalIdV1,
        variant: u32,
    },
}

#[derive(Clone, Debug)]
enum SemanticValueBindingV1 {
    Unit,
    Unmaterialized,
    Aggregate(Vec<SemanticValueBindingV1>),
    Enum {
        discriminant: ValueId,
        discriminant_ty: Type,
        semantic_type: SemanticTypeIdV1,
        variant: Option<u32>,
        payloads: BTreeMap<u32, Vec<SemanticValueBindingV1>>,
    },
    MathContext,
    CollectiveContext,
    WorkgroupLdsScope,
    MatrixContext,
    WaveLane {
        value: ValueId,
        wave: SemanticCurrentWaveV1,
    },
    MatrixFragment {
        values: Vec<(ValueId, Type)>,
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
        wave: SemanticCurrentWaveV1,
    },
    AccumulatorFragment {
        values: Vec<(ValueId, Type)>,
        contract: SemanticMfmaAccumulatorContractV1,
        wave: SemanticCurrentWaveV1,
    },
    Gfx950LdsTransposeTile {
        storage: ValueId,
        format: SemanticGfx950LdsTransposeFormatV1,
        state: SemanticGfx950LdsTransposeStateV1,
    },
    WorkgroupPipeline {
        storage: ValueId,
        pipeline: SemanticTypeIdV1,
        element: SemanticTypeIdV1,
        payload_binding: SemanticPromotedBindingV1,
        component_types: Box<[Type]>,
        packed_type: Type,
        buffers: u32,
        elements: u64,
        prefetch_distance: u32,
        alignment: u32,
    },
    Value {
        id: ValueId,
        ty: Type,
    },
    OptionPointer {
        present: ValueId,
        pointer: ValueId,
        pointer_ty: Type,
    },
    IndexWitness {
        id: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        disjoint: bool,
        availability: Option<SemanticCapabilityAvailabilityV1>,
    },
    OptionIndexWitness {
        present: ValueId,
        id: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticOptionAvailabilityV1,
    },
    GridLeader {
        availability: SemanticCapabilityAvailabilityV1,
    },
    ComponentWitness {
        raw: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticCapabilityAvailabilityV1,
    },
    OptionComponentWitness {
        present: ValueId,
        raw: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticOptionAvailabilityV1,
    },
    OptionGridLeader {
        present: ValueId,
        availability: SemanticOptionAvailabilityV1,
    },
}

fn project_enum_payload_field(
    selected_variant: u32,
    payloads: &BTreeMap<u32, Vec<SemanticValueBindingV1>>,
    field: u32,
) -> Result<SemanticValueBindingV1, &'static str> {
    let Some(fields) = payloads.get(&selected_variant) else {
        return Ok(SemanticValueBindingV1::Unmaterialized);
    };
    fields
        .get(field as usize)
        .cloned()
        .ok_or("enum payload field is unavailable in this block")
}

fn semantic_binding_kind_v1(binding: &SemanticValueBindingV1) -> &'static str {
    match binding {
        SemanticValueBindingV1::Unit => "unit",
        SemanticValueBindingV1::Unmaterialized => "unmaterialized enum payload",
        SemanticValueBindingV1::Aggregate(_) => "aggregate",
        SemanticValueBindingV1::Enum {
            variant: Some(_), ..
        } => "variant-refined enum",
        SemanticValueBindingV1::Enum { variant: None, .. } => "unrefined enum",
        SemanticValueBindingV1::MathContext => "math context",
        SemanticValueBindingV1::CollectiveContext => "collective context",
        SemanticValueBindingV1::WorkgroupLdsScope => "workgroup LDS scope",
        SemanticValueBindingV1::MatrixContext => "matrix context",
        SemanticValueBindingV1::WaveLane { .. } => "wave lane",
        SemanticValueBindingV1::MatrixFragment { .. } => "matrix fragment",
        SemanticValueBindingV1::AccumulatorFragment { .. } => "accumulator fragment",
        SemanticValueBindingV1::Gfx950LdsTransposeTile { .. } => "gfx950 LDS transpose tile",
        SemanticValueBindingV1::WorkgroupPipeline { .. } => "workgroup pipeline",
        SemanticValueBindingV1::Value { .. } => "ordinary value",
        SemanticValueBindingV1::OptionPointer { .. } => "optional pointer",
        SemanticValueBindingV1::IndexWitness { .. } => "index witness",
        SemanticValueBindingV1::OptionIndexWitness { .. } => "optional index witness",
        SemanticValueBindingV1::GridLeader { .. } => "grid leader",
        SemanticValueBindingV1::ComponentWitness { .. } => "component witness",
        SemanticValueBindingV1::OptionComponentWitness { .. } => "optional component witness",
        SemanticValueBindingV1::OptionGridLeader { .. } => "optional grid leader",
    }
}

fn semantic_binding_can_restore_from_unique_source_v1(binding: &SemanticValueBindingV1) -> bool {
    match binding {
        SemanticValueBindingV1::Unit
        | SemanticValueBindingV1::MathContext
        | SemanticValueBindingV1::CollectiveContext
        | SemanticValueBindingV1::MatrixContext
        | SemanticValueBindingV1::WaveLane { .. }
        | SemanticValueBindingV1::MatrixFragment { .. }
        | SemanticValueBindingV1::AccumulatorFragment { .. }
        | SemanticValueBindingV1::Gfx950LdsTransposeTile { .. }
        | SemanticValueBindingV1::WorkgroupPipeline { .. }
        | SemanticValueBindingV1::IndexWitness { .. }
        | SemanticValueBindingV1::GridLeader { .. }
        | SemanticValueBindingV1::ComponentWitness { .. } => true,
        SemanticValueBindingV1::Aggregate(fields) => fields
            .iter()
            .all(semantic_binding_can_restore_from_unique_source_v1),
        SemanticValueBindingV1::Value { .. } => true,
        SemanticValueBindingV1::Unmaterialized
        | SemanticValueBindingV1::Enum { .. }
        | SemanticValueBindingV1::OptionPointer { .. }
        | SemanticValueBindingV1::OptionIndexWitness { .. }
        | SemanticValueBindingV1::OptionComponentWitness { .. }
        | SemanticValueBindingV1::OptionGridLeader { .. }
        | SemanticValueBindingV1::WorkgroupLdsScope => false,
    }
}

fn reauthenticate_capabilities_from_enum_payload_v1(
    binding: &mut SemanticValueBindingV1,
    local: SemanticLocalIdV1,
    variant: u32,
) {
    let availability = SemanticCapabilityAvailabilityV1::EnumPayload { local, variant };
    match binding {
        SemanticValueBindingV1::Aggregate(fields) => {
            for field in fields {
                reauthenticate_capabilities_from_enum_payload_v1(field, local, variant);
            }
        }
        SemanticValueBindingV1::Enum {
            variant: Some(selected),
            payloads,
            ..
        } => {
            if let Some(fields) = payloads.get_mut(selected) {
                for field in fields {
                    reauthenticate_capabilities_from_enum_payload_v1(field, local, variant);
                }
            }
        }
        SemanticValueBindingV1::IndexWitness {
            availability: slot @ Some(_),
            ..
        } => *slot = Some(availability),
        SemanticValueBindingV1::GridLeader { availability: slot }
        | SemanticValueBindingV1::ComponentWitness {
            availability: slot, ..
        } => *slot = availability,
        SemanticValueBindingV1::Unit
        | SemanticValueBindingV1::Unmaterialized
        | SemanticValueBindingV1::Enum { .. }
        | SemanticValueBindingV1::MathContext
        | SemanticValueBindingV1::CollectiveContext
        | SemanticValueBindingV1::WorkgroupLdsScope
        | SemanticValueBindingV1::MatrixContext
        | SemanticValueBindingV1::WaveLane { .. }
        | SemanticValueBindingV1::MatrixFragment { .. }
        | SemanticValueBindingV1::AccumulatorFragment { .. }
        | SemanticValueBindingV1::Gfx950LdsTransposeTile { .. }
        | SemanticValueBindingV1::WorkgroupPipeline { .. }
        | SemanticValueBindingV1::Value { .. }
        | SemanticValueBindingV1::OptionPointer { .. }
        | SemanticValueBindingV1::IndexWitness {
            availability: None, ..
        }
        | SemanticValueBindingV1::OptionIndexWitness { .. }
        | SemanticValueBindingV1::OptionComponentWitness { .. }
        | SemanticValueBindingV1::OptionGridLeader { .. } => {}
    }
}

impl SemanticValueBindingV1 {
    fn value(&self) -> Result<(ValueId, Type), &'static str> {
        match self {
            Self::Value { id, ty } => Ok((*id, ty.clone())),
            Self::IndexWitness { id, .. } => Ok((*id, Type::INDEX)),
            Self::WaveLane { value, .. } => Ok((*value, Type::Scalar(ScalarType::U32))),
            Self::Unmaterialized => {
                Err("unmaterialized enum payload has no ordinary SSA representation")
            }
            Self::Unit
            | Self::Aggregate(_)
            | Self::Enum { .. }
            | Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::MatrixContext
            | Self::MatrixFragment { .. }
            | Self::AccumulatorFragment { .. }
            | Self::Gfx950LdsTransposeTile { .. }
            | Self::WorkgroupPipeline { .. }
            | Self::OptionPointer { .. }
            | Self::OptionIndexWitness { .. }
            | Self::ComponentWitness { .. }
            | Self::OptionComponentWitness { .. }
            | Self::GridLeader { .. }
            | Self::OptionGridLeader { .. } => {
                Err("aggregate or capability value requires a semantic projection")
            }
        }
    }

    fn values(&self) -> Result<Vec<(ValueId, Type)>, &'static str> {
        let mut values = Vec::new();
        self.append_values(&mut values)?;
        Ok(values)
    }

    fn append_values(&self, values: &mut Vec<(ValueId, Type)>) -> Result<(), &'static str> {
        match self {
            Self::Value { id, ty } => values.push((*id, ty.clone())),
            Self::IndexWitness { id, .. } => values.push((*id, Type::INDEX)),
            Self::WaveLane { value, .. } => {
                values.push((*value, Type::Scalar(ScalarType::U32)));
            }
            Self::Aggregate(fields) => {
                for field in fields {
                    field.append_values(values)?;
                }
            }
            Self::MatrixFragment {
                values: components, ..
            }
            | Self::AccumulatorFragment {
                values: components, ..
            } => {
                values.extend(components.iter().cloned());
            }
            Self::Gfx950LdsTransposeTile { storage, .. } => {
                values.push((*storage, gfx950_lds_transpose_pointer_type_v1()));
            }
            Self::Enum {
                discriminant,
                discriminant_ty,
                ..
            } => values.push((*discriminant, discriminant_ty.clone())),
            Self::Unit => {}
            Self::Unmaterialized => {
                return Err("unmaterialized enum payload has no ordinary SSA representation");
            }
            Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::MatrixContext
            | Self::WorkgroupPipeline { .. }
            | Self::OptionPointer { .. }
            | Self::OptionIndexWitness { .. }
            | Self::ComponentWitness { .. }
            | Self::OptionComponentWitness { .. }
            | Self::GridLeader { .. }
            | Self::OptionGridLeader { .. } => {
                return Err("capability value has no ordinary SSA representation");
            }
        }
        Ok(())
    }
}

fn require_components(
    block: SemanticBlockIdV1,
    values: Vec<(ValueId, Type)>,
    expected_type: Type,
    expected_count: usize,
    description: &'static str,
) -> Result<Vec<(ValueId, Type)>, ProductionSemanticKirErrorV1> {
    if values.len() != expected_count || values.iter().any(|(_, actual)| actual != &expected_type) {
        return Err(unsupported(0, Some(block.index()), None, description));
    }
    Ok(values)
}

fn require_single_u32_component(
    block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
    description: &'static str,
) -> Result<ValueId, ProductionSemanticKirErrorV1> {
    Ok(require_components(
        block,
        binding
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?,
        Type::Scalar(ScalarType::U32),
        1,
        description,
    )?[0]
        .0)
}

fn index_and_u64_are_transport_equivalent(actual: &Type, expected: &Type) -> bool {
    matches!(
        (actual, expected),
        (
            Type::Scalar(ScalarType::Index),
            Type::Scalar(ScalarType::U64)
        ) | (
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::Index)
        )
    )
}

fn require_current_wave_lane(
    block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
    expected_width: u32,
    description: &'static str,
) -> Result<(ValueId, SemanticCurrentWaveV1), ProductionSemanticKirErrorV1> {
    let SemanticValueBindingV1::WaveLane { value, wave } = binding else {
        return Err(unsupported(0, Some(block.index()), None, description));
    };
    if wave.width != expected_width {
        return Err(unsupported(0, Some(block.index()), None, description));
    }
    Ok((value, wave))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticMfmaOperandBasesV1<T> {
    minor: T,
    reduction: T,
}

fn semantic_mfma_operand_bases_v1<T>(
    role: fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1,
    first: T,
    second: T,
) -> SemanticMfmaOperandBasesV1<T> {
    use fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1;

    match role {
        SemanticMfmaOperandRoleV1::A => SemanticMfmaOperandBasesV1 {
            minor: first,
            reduction: second,
        },
        SemanticMfmaOperandRoleV1::B => SemanticMfmaOperandBasesV1 {
            minor: second,
            reduction: first,
        },
    }
}

struct SemanticFunctionLoweringV1<'a> {
    types: &'a [SemanticTypeDeclV1],
    callables: &'a [SemanticCallableDeclV1],
    function: &'a SemanticFunctionDeclV1,
    locals: Vec<Option<SemanticValueBindingV1>>,
    option_dominance: SemanticOptionDominanceV1,
    enum_payload_dominance: SemanticEnumPayloadDominanceV1,
    enum_payload_storage: BTreeMap<(u32, u32, u32), SemanticEnumPayloadFieldStorageV1>,
    enum_payload_sources: BTreeMap<(u32, u32, u32), SemanticEnumPayloadSourceV1>,
    enum_payload_requires_compile_time_custody: BTreeSet<(u32, u32, u32)>,
    enum_payload_compile_time_custody: BTreeMap<(u32, u32, u32), SemanticEnumPayloadCustodyV1>,
    enum_payload_allocas_emitted: bool,
    control_flow_ssa: SemanticControlFlowSsaPlanV1,
    workgroup_pipeline_contracts:
        BTreeMap<SemanticTypeIdV1, SemanticWorkgroupPipelineTypeContractV1>,
    promoted_enum_variant_by_block: BTreeMap<(u32, u32), u32>,
    block_parameters: BTreeMap<u32, BTreeMap<u32, Vec<ValueDef>>>,
    next_value: u32,
    assert_failure_block: Option<BlockId>,
    required_workgroup: Option<[u32; 3]>,
    infallible_asserts: BTreeSet<u32>,
    launch_rank: u8,
    max_operations: usize,
    emitted_operations: usize,
    emitted_u32_constants: BTreeMap<ValueId, u32>,
    emitted_u32_bitand_masks: BTreeMap<ValueId, u32>,
    authenticated_loop_induction_bounds: BTreeMap<(u32, u32), u128>,
    emitted_unsigned_exclusive_bounds: BTreeMap<ValueId, u128>,
}

struct SemanticParameterBindingsV1<'a> {
    declarations: &'a [(u32, usize, SemanticTypeIdV1)],
    values: &'a [ValueId],
    types: &'a [Type],
}

impl<'a> SemanticFunctionLoweringV1<'a> {
    fn new(
        types: &'a [SemanticTypeDeclV1],
        callables: &'a [SemanticCallableDeclV1],
        function: &'a SemanticFunctionDeclV1,
        parameters: SemanticParameterBindingsV1<'_>,
        assert_failure_block: Option<BlockId>,
        required_workgroup: Option<[u32; 3]>,
        infallible_asserts: BTreeSet<u32>,
        launch_rank: u8,
        authenticated_ranked_control: bool,
        max_operations: usize,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let mut locals = vec![None; function.locals().len()];
        let option_producers = semantic_option_producers_v1(function, callables)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        let option_dominance = SemanticOptionDominanceV1::analyze(function, &option_producers)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        let enum_payload_dominance = SemanticEnumPayloadDominanceV1::analyze(function, types)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        for ((_, local, _), (value, ty)) in parameters
            .declarations
            .iter()
            .zip(parameters.values.iter().zip(parameters.types))
        {
            locals[*local] = Some(SemanticValueBindingV1::Value {
                id: *value,
                ty: ty.clone(),
            });
        }
        let mut next_value = u32::try_from(function.locals().len())
            .map_err(|_| unsupported(0, None, None, "local count does not fit Kernel IR"))?;
        let control_flow_ssa = SemanticControlFlowSsaPlanV1::analyze(types, callables, function)?;
        let workgroup_pipeline_contracts = workgroup_pipeline_type_contracts_v1(
            types,
            callables,
            &control_flow_ssa.compiler_issued_bindings,
        )?;
        let authenticated_loop_induction_bounds = if authenticated_ranked_control {
            authenticated_loop_induction_bounds_v1(types, function)?
        } else {
            BTreeMap::new()
        };
        let promoted_enum_variant_by_block =
            analyze_promoted_enum_variants_v1(types, function, &control_flow_ssa)?;
        let mut block_parameters = BTreeMap::new();
        for block in 0..function.blocks().len() as u32 {
            if block == function.entry().index() {
                continue;
            }
            let mut parameters = BTreeMap::new();
            for local in control_flow_ssa.live_in(block) {
                let promoted = control_flow_ssa
                    .promoted
                    .get(local)
                    .expect("live-in local must be promoted");
                let mut components = Vec::with_capacity(promoted.kernel_types.len());
                for ty in promoted.kernel_types.iter().cloned() {
                    components.push(ValueDef::new(ValueId(next_value), ty));
                    next_value = next_value.checked_add(1).ok_or_else(|| {
                        unsupported(0, Some(block), None, "block-parameter identity overflow")
                    })?;
                }
                parameters.insert(*local, components);
            }
            block_parameters.insert(block, parameters);
        }
        let enum_payload_sources = plan_unique_enum_payload_sources_v1(function, &control_flow_ssa);
        let (enum_payload_storage, enum_payload_requires_compile_time_custody) =
            plan_enum_payload_storage_v1(
                types,
                function,
                &control_flow_ssa,
                &enum_payload_sources,
                &mut next_value,
            )?;
        Ok(Self {
            types,
            callables,
            function,
            locals,
            option_dominance,
            enum_payload_dominance,
            enum_payload_storage,
            enum_payload_sources,
            enum_payload_requires_compile_time_custody,
            enum_payload_compile_time_custody: BTreeMap::new(),
            enum_payload_allocas_emitted: false,
            control_flow_ssa,
            workgroup_pipeline_contracts,
            promoted_enum_variant_by_block,
            block_parameters,
            next_value,
            assert_failure_block,
            required_workgroup,
            infallible_asserts,
            launch_rank,
            max_operations,
            emitted_operations: 0,
            emitted_u32_constants: BTreeMap::new(),
            emitted_u32_bitand_masks: BTreeMap::new(),
            authenticated_loop_induction_bounds,
            emitted_unsigned_exclusive_bounds: BTreeMap::new(),
        })
    }

    fn begin_block(
        &mut self,
        block: SemanticBlockIdV1,
        target: &mut BasicBlock,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        for local in self.control_flow_ssa.promoted.keys() {
            let declaration = self.function.locals().get(*local as usize).ok_or_else(|| {
                unsupported(0, Some(block.index()), None, "promoted local is missing")
            })?;
            if block == self.function.entry()
                && matches!(declaration.role(), SemanticLocalRoleV1::Argument(_))
            {
                continue;
            }
            self.locals[*local as usize] = None;
        }
        if block == self.function.entry() {
            self.emit_enum_payload_allocas_v1(target)?;
            return Ok(());
        }
        let parameters = self
            .block_parameters
            .get(&block.index())
            .cloned()
            .ok_or_else(|| {
                unsupported(0, Some(block.index()), None, "block parameters are missing")
            })?;
        let parameter_locals = parameters.keys().copied().collect::<Vec<_>>();
        for (local, parameters) in parameters {
            let promoted = &self.control_flow_ssa.promoted[&local];
            self.locals[local as usize] = Some(promoted.binding.binding_from_transport(
                self.types,
                promoted.semantic_type,
                &parameters,
            )?);
            target.parameters.extend(parameters);
        }
        for local in parameter_locals {
            self.refine_enum_payload_at_block_v1(block, local, target)?;
        }
        Ok(())
    }

    fn emit_enum_payload_allocas_v1(
        &mut self,
        target: &mut BasicBlock,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if self.enum_payload_allocas_emitted {
            return Err(unsupported(
                0,
                Some(target.id.0),
                None,
                "enum payload storage allocas were emitted more than once",
            ));
        }
        let components = self
            .enum_payload_storage
            .values()
            .flat_map(|field| field.components.iter().cloned())
            .collect::<Vec<_>>();
        for component in components {
            let pointer_type = Type::pointer(
                component.kernel_type.clone(),
                AddressSpace::Private,
                AccessMode::ReadWrite,
            );
            self.push_operation(&mut target.operations, || {
                Operation::effect_free(
                    ValueDef::new(component.pointer, pointer_type),
                    OperationKind::Alloca {
                        element: component.kernel_type,
                        count: None,
                        address_space: AddressSpace::Private,
                        alignment: component.alignment,
                    },
                )
            })?;
        }
        self.enum_payload_allocas_emitted = true;
        Ok(())
    }

    fn refine_enum_payload_at_block_v1(
        &mut self,
        block: SemanticBlockIdV1,
        local: u32,
        target: &mut BasicBlock,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let Some(promoted) = self.control_flow_ssa.promoted.get(&local) else {
            return Ok(());
        };
        let Some(declaration) = self.types.get(promoted.semantic_type.index() as usize) else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "promoted enum type is missing",
            ));
        };
        let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
            return Ok(());
        };
        let mut selected = variants.iter().enumerate().filter_map(|(variant, _)| {
            let variant = variant as u32;
            self.enum_variant_is_available_v1(SemanticLocalIdV1::from_index(local), variant, block)
                .then_some(variant)
        });
        let Some(variant) = selected.next() else {
            return Ok(());
        };
        if selected.next().is_some() {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "multiple enum variants are available in one block",
            ));
        }
        let binding = self
            .locals
            .get_mut(local as usize)
            .and_then(Option::take)
            .ok_or(ProductionSemanticKirErrorV1::MissingLocalDefinition {
                function: 0,
                block: block.index(),
                statement: None,
                local,
            })?;
        let SemanticValueBindingV1::Enum {
            discriminant,
            discriminant_ty,
            semantic_type,
            ..
        } = binding
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "variant-refined local is not an enum SSA binding",
            ));
        };
        let variant_definition = variants.get(variant as usize).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "refined enum variant is missing",
            )
        })?;
        let mut restorations = Vec::with_capacity(variant_definition.fields().fields().len());
        for (field, _field_type) in variant_definition
            .fields()
            .fields()
            .iter()
            .copied()
            .enumerate()
        {
            let key = (local, variant, field as u32);
            let storage = self.enum_payload_storage.get(&key).cloned();
            let compile_time_custody = self
                .enum_payload_compile_time_custody
                .get(&key)
                .cloned()
                .filter(|custody| {
                    custody.source_block != block
                        && self
                            .enum_payload_dominance
                            .block_dominates(custody.source_block, block)
                        && semantic_binding_can_restore_from_unique_source_v1(&custody.binding)
                });
            if let Some(custody) = compile_time_custody {
                restorations.push(SemanticEnumPayloadRestoreV1::UniqueSource(custody.binding));
            } else if let Some(storage) = storage {
                restorations.push(SemanticEnumPayloadRestoreV1::PrivateStorage(storage));
            } else if self
                .enum_payload_requires_compile_time_custody
                .contains(&key)
            {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "non-storable enum payload source does not strictly dominate its refined use",
                ));
            } else {
                self.locals[local as usize] = Some(SemanticValueBindingV1::Enum {
                    discriminant,
                    discriminant_ty,
                    semantic_type,
                    variant: None,
                    payloads: BTreeMap::new(),
                });
                return Ok(());
            }
        }
        let mut fields = Vec::with_capacity(variant_definition.fields().fields().len());
        for (field_type, restoration) in variant_definition
            .fields()
            .fields()
            .iter()
            .copied()
            .zip(restorations)
        {
            match restoration {
                SemanticEnumPayloadRestoreV1::UniqueSource(source) => fields.push(source),
                SemanticEnumPayloadRestoreV1::PrivateStorage(storage) => {
                    if storage.semantic_type != field_type {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "refined enum payload storage type changed",
                        ));
                    }
                    let mut values = Vec::with_capacity(storage.components.len());
                    for component in storage.components.iter() {
                        let value = self
                            .emit(
                                &mut target.operations,
                                component.kernel_type.clone(),
                                OperationKind::Load {
                                    pointer: component.pointer,
                                    access: MemoryAccess::new(
                                        AddressSpace::Private,
                                        component.alignment,
                                    ),
                                },
                            )?
                            .value()
                            .expect("enum payload component load returns one value");
                        values.push(ValueDef::new(value.0, value.1));
                    }
                    fields.push(match storage.exact_enum_variant {
                        Some(exact_variant) => binding_from_exact_enum_value_defs_v1(
                            self.types,
                            field_type,
                            exact_variant,
                            &values,
                        )?,
                        None => match storage.compiler_issued_binding {
                            Some(binding) => {
                                binding.binding_from_transport(self.types, field_type, &values)?
                            }
                            None => binding_from_value_defs(self.types, field_type, &values)?,
                        },
                    });
                }
            }
        }
        for field in &mut fields {
            reauthenticate_capabilities_from_enum_payload_v1(
                field,
                SemanticLocalIdV1::from_index(local),
                variant,
            );
        }
        self.locals[local as usize] = Some(SemanticValueBindingV1::Enum {
            discriminant,
            discriminant_ty,
            semantic_type,
            variant: Some(variant),
            payloads: BTreeMap::from([(variant, fields)]),
        });
        Ok(())
    }

    fn enum_variant_is_available_v1(
        &self,
        local: SemanticLocalIdV1,
        variant: u32,
        block: SemanticBlockIdV1,
    ) -> bool {
        self.promoted_enum_variant_by_block
            .get(&(block.index(), local.index()))
            .is_some_and(|available| *available == variant)
            || self
                .enum_payload_dominance
                .availability(local, variant)
                .is_some_and(|availability| self.enum_payload_dominance.allows(availability, block))
    }

    fn edge_arguments(
        &mut self,
        block: SemanticBlockIdV1,
        target: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<Vec<ValueId>, ProductionSemanticKirErrorV1> {
        let mut arguments = Vec::new();
        let live_in_count = self.control_flow_ssa.live_in(target.index()).len();
        for live_in_ordinal in 0..live_in_count {
            let local = self.control_flow_ssa.live_in(target.index())[live_in_ordinal];
            let (values, expected_count) = {
                let binding = self
                    .locals
                    .get(local as usize)
                    .and_then(Option::as_ref)
                    .ok_or(ProductionSemanticKirErrorV1::MissingLocalDefinition {
                        function: 0,
                        block: block.index(),
                        statement: None,
                        local,
                    })?;
                let promoted = &self.control_flow_ssa.promoted[&local];
                let values = promoted
                    .binding
                    .transport_values(binding)
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                (values, promoted.kernel_types.len())
            };
            if values.len() != expected_count {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "promoted aggregate changed its SSA component types",
                ));
            }
            for (component, (value, actual)) in values.into_iter().enumerate() {
                let expected =
                    self.control_flow_ssa.promoted[&local].kernel_types[component].clone();
                let value = self.coerce_transport_value_v1(
                    operations,
                    block,
                    None,
                    value,
                    actual,
                    expected,
                    "promoted aggregate changed its SSA component types",
                )?;
                arguments.push(value);
            }
        }
        Ok(arguments)
    }

    fn lower_statement(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        kind: &SemanticStatementKindV1,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        match kind {
            SemanticStatementKindV1::Assign(assignment) => {
                let checked_checkpoint = matches!(
                    assignment.value().kind(),
                    SemanticRvalueKindV1::CheckedBinary(_)
                )
                .then_some((
                    operations.len(),
                    self.next_value,
                    self.emitted_operations,
                ));
                let result = self.lower_rvalue(
                    block,
                    statement,
                    assignment.value().result_type(),
                    assignment.value().kind(),
                    operations,
                );
                let result = match result {
                    Ok(value) => self.assign_place(
                        block,
                        statement,
                        assignment.destination(),
                        value,
                        SemanticVolatilityV1::NonVolatile,
                        operations,
                    ),
                    Err(error) => Err(error),
                };
                if result.is_err()
                    && let Some((operation_count, next_value, emitted_operations)) =
                        checked_checkpoint
                {
                    operations.truncate(operation_count);
                    self.next_value = next_value;
                    self.emitted_operations = emitted_operations;
                }
                result
            }
            SemanticStatementKindV1::Store(store) if store.atomic().is_none() => {
                let value = self.lower_operand(block, statement, store.value(), operations)?;
                self.assign_place(
                    block,
                    statement,
                    store.destination(),
                    value,
                    store.volatility(),
                    operations,
                )
            }
            SemanticStatementKindV1::StorageLive(local)
            | SemanticStatementKindV1::StorageDead(local) => {
                self.require_local(block, statement, local.index())?;
                Ok(())
            }
            SemanticStatementKindV1::Assume(condition) => {
                let _ = self.lower_operand(block, statement, condition, operations)?;
                Ok(())
            }
            SemanticStatementKindV1::AtomicRmw(atomic) => {
                self.lower_atomic_rmw(block, statement, atomic, operations)
            }
            SemanticStatementKindV1::Nop => Ok(()),
            _ => Err(unsupported(
                0,
                Some(block.index()),
                statement,
                unsupported_statement_detail(kind)
                    .unwrap_or("semantic statement has no exact Kernel IR lowering rule"),
            )),
        }
    }

    fn lower_atomic_rmw(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        atomic: &SemanticAtomicRmwV1,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let (pointer, pointer_ty) = self
            .resolve_place(block, statement, atomic.address())?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
        let Type::Pointer(pointer_ty) = pointer_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic-rmw address is not a lowered pointer",
            ));
        };
        if pointer_ty.access != AccessMode::ReadWrite {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic-rmw address is not writable",
            ));
        }
        let pointee = (*pointer_ty.pointee).clone();
        let scalar = pointee.as_scalar().ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic-rmw pointee is not a physical scalar",
            )
        })?;
        let kind = lower_atomic_rmw_kind(atomic.operation(), scalar).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic-rmw operation has no exact Kernel IR operation",
            )
        })?;
        let scope = lower_atomic_scope(atomic.access().scope()).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic scope has no exact Kernel IR scope",
            )
        })?;
        let value = self.lower_operand(block, statement, atomic.value(), operations)?;
        let (value, value_ty) = value
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
        if value_ty != pointee
            || lower_memory_element_type(self.types, atomic.destination().ty())? != pointee
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic atomic-rmw value or result type differs from its pointee",
            ));
        }
        let access =
            memory_access_for_type(self.types, atomic.address().ty(), pointer_ty.address_space)?;
        let result = self.emit(
            operations,
            pointee,
            OperationKind::Atomic(Atomic {
                kind,
                pointer,
                value: Some(value),
                compare: None,
                access,
                scope,
                ordering: lower_atomic_ordering(atomic.access().ordering()),
                failure_ordering: None,
            }),
        )?;
        self.bind_destination(block, statement, atomic.destination(), result)
    }

    fn lower_rvalue(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        result_type: SemanticTypeIdV1,
        value: &SemanticRvalueKindV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        match value {
            SemanticRvalueKindV1::Use(operand) => {
                self.lower_operand(block, statement, operand, operations)
            }
            SemanticRvalueKindV1::Borrow { kind, place }
                if *kind == fe2o3_mir_model::semantic_mir_v1::SemanticBorrowKindV1::Mutable
                    && place.projections().is_empty()
                    && authenticated_workgroup_lds_scope_type_v1(
                        self.types,
                        self.callables,
                        place.ty(),
                    ) =>
            {
                Ok(SemanticValueBindingV1::WorkgroupLdsScope)
            }
            SemanticRvalueKindV1::Borrow { place, .. }
            | SemanticRvalueKindV1::AddressOf { place, .. } => {
                if place.projections().is_empty() {
                    let local = self.require_local(block, statement, place.local().index())?;
                    let declaration = &self.function.locals()[local];
                    if self.locals[local].is_none()
                        && declaration.role() == SemanticLocalRoleV1::Temporary
                        && !self.local_has_direct_definition_v1(place.local())
                        && let Some(binding) =
                            self.reauthenticate_compiler_capability_zst(block, place.ty())?
                    {
                        return Ok(binding);
                    }
                }
                if place.projections().iter().any(|projection| {
                    matches!(projection.kind(), SemanticProjectionKindV1::Index(_))
                }) {
                    self.lower_indexed_place_address(block, statement, place, operations)
                } else {
                    self.resolve_place(block, statement, place)
                }
            }
            SemanticRvalueKindV1::Discriminant(place) => {
                match self.resolve_place(block, statement, place)? {
                    SemanticValueBindingV1::Enum {
                        discriminant,
                        discriminant_ty,
                        ..
                    } => Ok(SemanticValueBindingV1::Value {
                        id: discriminant,
                        ty: discriminant_ty,
                    }),
                    SemanticValueBindingV1::OptionPointer { present, .. }
                    | SemanticValueBindingV1::OptionIndexWitness { present, .. }
                    | SemanticValueBindingV1::OptionComponentWitness { present, .. }
                    | SemanticValueBindingV1::OptionGridLeader { present, .. } => {
                        let target = lower_scalar_type(self.types, result_type)?;
                        if target == Type::BOOL {
                            Ok(SemanticValueBindingV1::Value {
                                id: present,
                                ty: Type::BOOL,
                            })
                        } else if target.as_scalar().is_some_and(ScalarType::is_integer) {
                            self.emit(
                                operations,
                                target.clone(),
                                OperationKind::Cast {
                                    kind: CastKind::ZeroExtend,
                                    value: present,
                                    to: target,
                                },
                            )
                        } else {
                            Err(unsupported(
                                0,
                                Some(block.index()),
                                statement,
                                "semantic option discriminant is not integer-valued",
                            ))
                        }
                    }
                    SemanticValueBindingV1::Unit
                    | SemanticValueBindingV1::Unmaterialized
                    | SemanticValueBindingV1::Aggregate(_)
                    | SemanticValueBindingV1::MathContext
                    | SemanticValueBindingV1::CollectiveContext
                    | SemanticValueBindingV1::WorkgroupLdsScope
                    | SemanticValueBindingV1::MatrixContext
                    | SemanticValueBindingV1::WaveLane { .. }
                    | SemanticValueBindingV1::MatrixFragment { .. }
                    | SemanticValueBindingV1::AccumulatorFragment { .. }
                    | SemanticValueBindingV1::Gfx950LdsTransposeTile { .. }
                    | SemanticValueBindingV1::WorkgroupPipeline { .. }
                    | SemanticValueBindingV1::Value { .. }
                    | SemanticValueBindingV1::IndexWitness { .. }
                    | SemanticValueBindingV1::ComponentWitness { .. }
                    | SemanticValueBindingV1::GridLeader { .. } => Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic discriminant source is not a lowered option",
                    )),
                }
            }
            SemanticRvalueKindV1::Length(place) => {
                let (slice, ty) = self
                    .resolve_place(block, statement, place)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                if !matches!(ty, Type::Slice(_)) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic length source is not a lowered slice",
                    ));
                }
                self.emit(
                    operations,
                    Type::INDEX,
                    OperationKind::SliceLength { slice },
                )
            }
            SemanticRvalueKindV1::Unary { operation, operand } => {
                let input = self.lower_operand(block, statement, operand, operations)?;
                let (input, input_ty) = input
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                match operation {
                    SemanticUnaryOpV1::Not => self.emit(
                        operations,
                        input_ty,
                        OperationKind::Unary {
                            op: UnaryOp::Not,
                            operand: input,
                        },
                    ),
                    SemanticUnaryOpV1::Negate => self.emit(
                        operations,
                        input_ty,
                        OperationKind::Unary {
                            op: UnaryOp::Negate,
                            operand: input,
                        },
                    ),
                    SemanticUnaryOpV1::PointerMetadata if matches!(input_ty, Type::Slice(_)) => {
                        self.emit(
                            operations,
                            Type::INDEX,
                            OperationKind::SliceLength { slice: input },
                        )
                    }
                    SemanticUnaryOpV1::PointerMetadata => Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "pointer metadata is available only for lowered slices",
                    )),
                }
            }
            SemanticRvalueKindV1::Binary {
                operation,
                left,
                right,
            } => {
                let semantic_left_type = semantic_operand_type(left);
                let semantic_right_type = semantic_operand_type(right);
                let semantic_operands_match = semantic_left_type == semantic_right_type;
                let semantic_lowered_type = semantic_operands_match
                    .then(|| lower_scalar_type(self.types, semantic_left_type))
                    .transpose()?;
                let canonical_left = semantic_lowered_type
                    .as_ref()
                    .and_then(|ty| canonical_index_constant_v1(left, ty));
                let canonical_right = semantic_lowered_type
                    .as_ref()
                    .and_then(|ty| canonical_index_constant_v1(right, ty));
                let canonicalize_left = canonical_left.is_some()
                    && canonical_right.is_none()
                    && self.operand_transport_type(block, statement, right)? == Some(Type::INDEX);
                let canonicalize_right = canonical_right.is_some()
                    && canonical_left.is_none()
                    && self.operand_transport_type(block, statement, left)? == Some(Type::INDEX);
                let left = if canonicalize_left {
                    self.emit(
                        operations,
                        Type::INDEX,
                        OperationKind::Constant(canonical_left.expect("checked above")),
                    )?
                } else {
                    self.lower_operand(block, statement, left, operations)?
                };
                let (mut left, mut left_ty) = left
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                let canonical_shift_right = if !semantic_operands_match
                    && matches!(
                        operation,
                        SemanticBinaryOpV1::ShiftLeft | SemanticBinaryOpV1::ShiftRight
                    ) {
                    let semantic_right_lowered_type =
                        lower_scalar_type(self.types, semantic_right_type)?;
                    canonical_shift_rhs_constant_v1(
                        *operation,
                        right,
                        &semantic_right_lowered_type,
                        &left_ty,
                    )
                } else {
                    None
                };
                let right = if let Some(constant) = canonical_shift_right {
                    self.emit(
                        operations,
                        left_ty.clone(),
                        OperationKind::Constant(constant),
                    )?
                } else if canonicalize_right {
                    self.emit(
                        operations,
                        Type::INDEX,
                        OperationKind::Constant(canonical_right.expect("checked above")),
                    )?
                } else {
                    self.lower_operand(block, statement, right, operations)?
                };
                let (mut right, mut right_ty) = right
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                if let Some((convert_left, coercion)) = index_binary_coercion_v1(
                    semantic_operands_match,
                    left,
                    &left_ty,
                    right,
                    &right_ty,
                ) {
                    let converted = self.emit(operations, Type::INDEX, coercion)?;
                    let (converted, converted_ty) = converted
                        .value()
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                    if convert_left {
                        left = converted;
                        left_ty = converted_ty;
                    } else {
                        right = converted;
                        right_ty = converted_ty;
                    }
                }
                if left_ty != right_ty {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic binary operand types differ",
                    ));
                }
                if matches!(
                    operation,
                    SemanticBinaryOpV1::ShiftLeft | SemanticBinaryOpV1::ShiftRight
                ) && !left_ty.as_scalar().is_some_and(ScalarType::is_integer)
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic shift operands are not integral scalars",
                    ));
                }
                if let Some(predicate) = lower_compare(*operation) {
                    self.emit(
                        operations,
                        Type::BOOL,
                        OperationKind::Compare {
                            predicate,
                            lhs: left,
                            rhs: right,
                        },
                    )
                } else if let Some(operation) = lower_binary(*operation) {
                    self.emit(
                        operations,
                        left_ty,
                        OperationKind::Binary {
                            op: operation,
                            lhs: left,
                            rhs: right,
                        },
                    )
                } else {
                    Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic pointer offset requires an explicit GEP rule",
                    ))
                }
            }
            SemanticRvalueKindV1::CheckedBinary(checked) => {
                let semantic_operand_ty = semantic_operand_type(checked.left());
                if semantic_operand_ty != semantic_operand_type(checked.right()) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic checked arithmetic operand types differ",
                    ));
                }
                let operand_type =
                    checked_binary_result_type(self.types, semantic_operand_ty, result_type)
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                let left = self.lower_operand(block, statement, checked.left(), operations)?;
                let (left, left_type) = self.normalize_checked_operand(
                    block,
                    statement,
                    left,
                    &operand_type,
                    operations,
                )?;
                let right = self.lower_operand(block, statement, checked.right(), operations)?;
                let (right, right_type) = self.normalize_checked_operand(
                    block,
                    statement,
                    right,
                    &operand_type,
                    operations,
                )?;
                if left_type != right_type {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "lowered checked arithmetic operand types differ",
                    ));
                }
                self.emit_checked_binary(
                    operations,
                    left_type,
                    lower_checked_binary(checked.operation()),
                    left,
                    right,
                )
            }
            SemanticRvalueKindV1::UncheckedBinary(unchecked) => {
                let operation = match unchecked.operation() {
                    SemanticUncheckedBinaryOpV1::Add => SemanticBinaryOpV1::Add,
                    SemanticUncheckedBinaryOpV1::Subtract => SemanticBinaryOpV1::Subtract,
                    SemanticUncheckedBinaryOpV1::Multiply => SemanticBinaryOpV1::Multiply,
                };
                let checked_by_admission = SemanticRvalueKindV1::Binary {
                    operation,
                    left: unchecked.left().clone(),
                    right: unchecked.right().clone(),
                };
                self.lower_rvalue(
                    block,
                    statement,
                    result_type,
                    &checked_by_admission,
                    operations,
                )
            }
            SemanticRvalueKindV1::Cast { kind, operand } => {
                let (input, input_ty) = self
                    .lower_operand(block, statement, operand, operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                let target = match self
                    .types
                    .get(result_type.index() as usize)
                    .map(SemanticTypeDeclV1::shape)
                {
                    Some(SemanticTypeShapeV1::Pointer(_)) => {
                        lower_parameter_type(self.types, &[], result_type)?
                    }
                    _ => lower_scalar_type(self.types, result_type)?,
                };
                if input_ty == target {
                    return Ok(SemanticValueBindingV1::Value {
                        id: input,
                        ty: input_ty,
                    });
                }
                let Some(path) = lower_cast_path(*kind, &input_ty, &target) else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic cast has no exact Kernel IR cast rule",
                    ));
                };
                let mut input = input;
                let mut result = SemanticValueBindingV1::Value {
                    id: input,
                    ty: input_ty,
                };
                for (kind, target) in path.into_iter().flatten() {
                    let target = Type::Scalar(target);
                    result = self.emit(
                        operations,
                        target.clone(),
                        OperationKind::Cast {
                            kind,
                            value: input,
                            to: target,
                        },
                    )?;
                    input = result
                        .value()
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?
                        .0;
                }
                Ok(result)
            }
            SemanticRvalueKindV1::Load(load) if load.atomic().is_none() => {
                let (pointer, pointer_ty) = self
                    .resolve_place(block, statement, load.source())?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                let Type::Pointer(pointer_type) = pointer_ty else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic load source is not a lowered pointer",
                    ));
                };
                let mut access = memory_access_for_type(
                    self.types,
                    load.source().ty(),
                    pointer_type.address_space,
                )?;
                access.volatile = load.volatility() == SemanticVolatilityV1::Volatile;
                self.emit(
                    operations,
                    (*pointer_type.pointee).clone(),
                    OperationKind::Load { pointer, access },
                )
            }
            SemanticRvalueKindV1::Aggregate(aggregate)
                if matches!(aggregate.kind(), SemanticAggregateKindV1::EnumVariant(_)) =>
            {
                let SemanticAggregateKindV1::EnumVariant(variant) = aggregate.kind() else {
                    unreachable!("guard requires an enum aggregate")
                };
                let (discriminant_type, variants) = semantic_enum_shape(self.types, result_type)?;
                let selected = variants.get(*variant as usize).ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic enum aggregate variant is out of range",
                    )
                })?;
                let discriminant_ty = lower_scalar_type(self.types, discriminant_type)?;
                let discriminant = self
                    .emit(
                        operations,
                        discriminant_ty.clone(),
                        OperationKind::Constant(integer_constant(
                            &discriminant_ty,
                            selected.discriminant(),
                        )?),
                    )?
                    .value()
                    .expect("emitted enum discriminant")
                    .0;
                let mut fields = Vec::with_capacity(aggregate.operands().len());
                for operand in aggregate.operands() {
                    fields.push(self.lower_operand(block, statement, operand, operations)?);
                }
                Ok(SemanticValueBindingV1::Enum {
                    discriminant,
                    discriminant_ty,
                    semantic_type: result_type,
                    variant: Some(*variant),
                    payloads: BTreeMap::from([(*variant, fields)]),
                })
            }
            SemanticRvalueKindV1::Aggregate(aggregate)
                if matches!(
                    self.types[result_type.index() as usize].shape(),
                    SemanticTypeShapeV1::Array { .. }
                        | SemanticTypeShapeV1::Tuple(_)
                        | SemanticTypeShapeV1::Aggregate(_)
                ) =>
            {
                let mut fields = Vec::with_capacity(aggregate.operands().len());
                for operand in aggregate.operands() {
                    fields.push(self.lower_operand(block, statement, operand, operations)?);
                }
                if let Some(binding) =
                    self.reauthenticate_compiler_capability_zst(block, result_type)?
                {
                    return Ok(binding);
                }
                Ok(SemanticValueBindingV1::Aggregate(fields))
            }
            _ => Err(unsupported(
                0,
                Some(block.index()),
                statement,
                unsupported_rvalue_detail(value),
            )),
        }
    }

    fn lower_operand(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        operand: &SemanticOperandV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        match operand {
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => {
                let binding = if place.projections().iter().any(|projection| {
                    matches!(projection.kind(), SemanticProjectionKindV1::Index(_))
                }) {
                    self.lower_indexed_place_address(block, statement, place, operations)?
                } else {
                    self.resolve_place(block, statement, place)?
                };
                if !place
                    .projections()
                    .iter()
                    .any(|projection| projection.kind() == SemanticProjectionKindV1::Dereference)
                {
                    return Ok(binding);
                }
                if matches!(binding, SemanticValueBindingV1::WorkgroupPipeline { .. }) {
                    return Ok(binding);
                }
                let (pointer, pointer_ty) = binding
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                let Type::Pointer(pointer_type) = pointer_ty else {
                    return Ok(SemanticValueBindingV1::Value {
                        id: pointer,
                        ty: pointer_ty,
                    });
                };
                let mut access =
                    memory_access_for_type(self.types, place.ty(), pointer_type.address_space)?;
                access.volatile = false;
                self.emit(
                    operations,
                    (*pointer_type.pointee).clone(),
                    OperationKind::Load { pointer, access },
                )
            }
            SemanticOperandV1::Constant(constant) => {
                if matches!(constant.value(), SemanticConstantValueV1::ZeroSized)
                    && let Some(binding) =
                        self.reauthenticate_compiler_capability_zst(block, constant.ty())?
                {
                    return Ok(binding);
                }
                if matches!(constant.value(), SemanticConstantValueV1::ZeroSized) {
                    let declaration =
                        self.types
                            .get(constant.ty().index() as usize)
                            .ok_or_else(|| {
                                unsupported(
                                    0,
                                    Some(block.index()),
                                    statement,
                                    "zero-sized constant type is missing",
                                )
                            })?;
                    if declaration.layout().size_bytes() != Some(0)
                        || declaration.layout().is_uninhabited()
                    {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "zero-sized constant lacks an exact inhabited zero-sized layout",
                        ));
                    }
                    let mut structural_nodes = 0;
                    return self.lower_constant_bytes(
                        block,
                        statement,
                        constant.ty(),
                        &[],
                        0,
                        &mut structural_nodes,
                        operations,
                    );
                }
                if let SemanticConstantValueV1::Scalar(value) = constant.value()
                    && matches!(
                        self.types[constant.ty().index() as usize].shape(),
                        SemanticTypeShapeV1::Enum { .. }
                    )
                {
                    let variant =
                        semantic_scalar_enum_variant_v1(self.types, constant.ty(), *value)
                            .ok_or_else(|| {
                                unsupported(
                                    0,
                                    Some(block.index()),
                                    statement,
                                    "scalar enum constant has no admitted logical variant",
                                )
                            })?;
                    let (discriminant_type, variants) =
                        semantic_enum_shape(self.types, constant.ty())?;
                    let logical = variants[variant as usize].discriminant();
                    let discriminant_ty = lower_scalar_type(self.types, discriminant_type)?;
                    let discriminant = self
                        .emit(
                            operations,
                            discriminant_ty.clone(),
                            OperationKind::Constant(integer_constant(&discriminant_ty, logical)?),
                        )?
                        .value()
                        .expect("emitted enum discriminant")
                        .0;
                    return Ok(SemanticValueBindingV1::Enum {
                        discriminant,
                        discriminant_ty,
                        semantic_type: constant.ty(),
                        variant: Some(variant),
                        payloads: BTreeMap::from([(variant, Vec::new())]),
                    });
                }
                if let SemanticConstantValueV1::Bytes(bytes) = constant.value() {
                    let mut structural_nodes = 0;
                    return self.lower_constant_bytes(
                        block,
                        statement,
                        constant.ty(),
                        bytes.as_bytes(),
                        0,
                        &mut structural_nodes,
                        operations,
                    );
                }
                let ty = lower_scalar_type(self.types, constant.ty())?;
                let SemanticConstantValueV1::Scalar(value) = constant.value() else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "semantic constant is not a scalar",
                    ));
                };
                self.emit(
                    operations,
                    ty.clone(),
                    OperationKind::Constant(lower_constant(ty, *value)?),
                )
            }
        }
    }

    fn local_has_direct_definition_v1(&self, local: SemanticLocalIdV1) -> bool {
        let is_direct =
            |place: &SemanticPlaceV1| place.local() == local && place.projections().is_empty();
        self.function.blocks().iter().any(|block| {
            block.statements().iter().any(|statement| match statement.kind() {
                SemanticStatementKindV1::Assign(assignment) => {
                    is_direct(assignment.destination())
                }
                SemanticStatementKindV1::Store(store) => is_direct(store.destination()),
                SemanticStatementKindV1::AtomicRmw(atomic) => {
                    is_direct(atomic.destination())
                }
                SemanticStatementKindV1::AtomicCompareExchange(atomic) => {
                    is_direct(atomic.destination())
                }
                SemanticStatementKindV1::SetDiscriminant { place, .. }
                | SemanticStatementKindV1::Deinitialize(place) => is_direct(place),
                SemanticStatementKindV1::StorageLive(_)
                | SemanticStatementKindV1::StorageDead(_)
                | SemanticStatementKindV1::Assume(_)
                | SemanticStatementKindV1::Nop => false,
            }) || matches!(
                block.terminator().kind(),
                SemanticTerminatorKindV1::Call(call)
                    if call.destination().is_some_and(|destination| is_direct(destination.place()))
            )
        })
    }

    fn reauthenticate_compiler_capability_zst(
        &self,
        block: SemanticBlockIdV1,
        ty: SemanticTypeIdV1,
    ) -> Result<Option<SemanticValueBindingV1>, ProductionSemanticKirErrorV1> {
        let is_grid_leader = self.callables.iter().any(|callable| {
            matches!(
                callable,
                SemanticCallableDeclV1::CompilerIntrinsic {
                    operation: SemanticCompilerIntrinsicOperationV1::GridLeaderCurrent {
                        grid_leader,
                    },
                    ..
                } if *grid_leader == ty
            )
        });
        if !is_grid_leader {
            return Ok(None);
        }

        let mut candidates = Vec::new();
        for source in self.function.blocks() {
            let SemanticTerminatorKindV1::Call(call) = source.terminator().kind() else {
                continue;
            };
            let Some(SemanticCallableDeclV1::CompilerIntrinsic {
                operation: SemanticCompilerIntrinsicOperationV1::GridLeaderCurrent { grid_leader },
                ..
            }) = self.callables.get(call.callee().index() as usize)
            else {
                continue;
            };
            if *grid_leader != ty {
                continue;
            }
            let destination = call.destination().ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "grid-leader producer has no Option destination",
                )
            })?;
            let availability = self
                .option_dominance
                .availability(destination.place().local())
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "grid-leader producer lacks authenticated Option dominance",
                    )
                })?;
            if self.option_dominance.allows(availability, block)
                && !candidates.contains(&availability)
            {
                candidates.push(availability);
            }
        }
        match candidates.as_slice() {
            [availability] => Ok(Some(SemanticValueBindingV1::GridLeader {
                availability: SemanticCapabilityAvailabilityV1::Option(*availability),
            })),
            [] => Err(unsupported(
                0,
                Some(block.index()),
                None,
                "grid-leader ZST constant is outside its authenticated Some region",
            )),
            _ => Err(unsupported(
                0,
                Some(block.index()),
                None,
                "grid-leader ZST constant has ambiguous authenticated producers",
            )),
        }
    }

    fn operand_transport_type(
        &self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        operand: &SemanticOperandV1,
    ) -> Result<Option<Type>, ProductionSemanticKirErrorV1> {
        let (SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) = operand else {
            return Ok(None);
        };
        if place.projections().iter().any(|projection| {
            matches!(
                projection.kind(),
                SemanticProjectionKindV1::Index(_) | SemanticProjectionKindV1::Dereference
            )
        }) {
            return Ok(None);
        }
        Ok(self
            .resolve_place(block, statement, place)?
            .value()
            .ok()
            .map(|(_, ty)| ty))
    }

    fn normalize_checked_operand(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        binding: SemanticValueBindingV1,
        expected_type: &Type,
        operations: &mut Vec<Operation>,
    ) -> Result<(ValueId, Type), ProductionSemanticKirErrorV1> {
        let (value, actual_type) = binding
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
        if &actual_type == expected_type {
            return Ok((value, actual_type));
        }
        if actual_type == Type::INDEX && *expected_type == Type::Scalar(ScalarType::U64) {
            return self
                .emit(
                    operations,
                    expected_type.clone(),
                    OperationKind::Cast {
                        kind: CastKind::Bitcast,
                        value,
                        to: expected_type.clone(),
                    },
                )?
                .value()
                .map_err(|detail| unsupported(0, Some(block.index()), statement, detail));
        }
        Err(unsupported(
            0,
            Some(block.index()),
            statement,
            "checked arithmetic operand has no exact plain-integer representation",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_constant_bytes(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        ty: SemanticTypeIdV1,
        bytes: &[u8],
        base_offset: u64,
        structural_nodes: &mut usize,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        *structural_nodes = structural_nodes.checked_add(1).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "constant aggregate exceeds the structural limit",
            )
        })?;
        if *structural_nodes > MAX_SSA_VALUE_COMPONENTS_V1 {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "constant aggregate exceeds the structural limit",
            ));
        }

        let declaration = self
            .types
            .get(ty.index() as usize)
            .ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    statement,
                    "constant type is missing",
                )
            })?
            .clone();
        let layout = declaration.layout();
        let size = layout.size_bytes().ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "constant type has no fixed-size Rust layout",
            )
        })?;
        checked_constant_range_v1(bytes, base_offset, size).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "constant bytes are truncated for their Rust layout",
            )
        })?;

        match declaration.shape() {
            SemanticTypeShapeV1::Unit => Ok(SemanticValueBindingV1::Unit),
            SemanticTypeShapeV1::Scalar(_) | SemanticTypeShapeV1::ValidityScalar(_) => {
                let size = u8::try_from(size).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "scalar constant exceeds the supported width",
                    )
                })?;
                let bits = read_constant_bits_v1(bytes, base_offset, u64::from(size)).ok_or_else(
                    || {
                        unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "scalar constant bytes are truncated",
                        )
                    },
                )?;
                let value = SemanticScalarValueV1::new(bits, size).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "scalar constant has an invalid Rust representation",
                    )
                })?;
                let lowered_ty = lower_scalar_type(self.types, ty)?;
                self.emit(
                    operations,
                    lowered_ty.clone(),
                    OperationKind::Constant(lower_constant(lowered_ty, value)?),
                )
            }
            SemanticTypeShapeV1::Array { element, length } => {
                let SemanticFieldsShapeV1::Array {
                    stride_bytes,
                    count,
                } = layout.fields()
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "array constant has no exact Rust array layout",
                    ));
                };
                if count != length {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "array constant shape disagrees with its Rust layout",
                    ));
                }
                let length = usize::try_from(*length).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "array constant length exceeds the structural limit",
                    )
                })?;
                let mut fields = Vec::new();
                fields.try_reserve_exact(length).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "array constant length exceeds available capacity",
                    )
                })?;
                for index in 0..length {
                    let field_offset = u64::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_mul(*stride_bytes))
                        .and_then(|offset| base_offset.checked_add(offset))
                        .ok_or_else(|| {
                            unsupported(
                                0,
                                Some(block.index()),
                                statement,
                                "array constant field offset overflows",
                            )
                        })?;
                    fields.push(self.lower_constant_bytes(
                        block,
                        statement,
                        *element,
                        bytes,
                        field_offset,
                        structural_nodes,
                        operations,
                    )?);
                }
                Ok(SemanticValueBindingV1::Aggregate(fields))
            }
            SemanticTypeShapeV1::Tuple(field_types)
            | SemanticTypeShapeV1::Aggregate(field_types) => {
                let SemanticTypeLayoutDetailsV1::Aggregate(aggregate) = layout.details() else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "aggregate constant has no exact Rust field layout",
                    ));
                };
                self.lower_constant_fields(
                    block,
                    statement,
                    field_types.fields(),
                    aggregate.field_offsets(),
                    bytes,
                    base_offset,
                    structural_nodes,
                    operations,
                )
                .map(SemanticValueBindingV1::Aggregate)
            }
            SemanticTypeShapeV1::Enum {
                discriminant,
                variants,
            } => {
                let (variant_index, offsets) = match layout.variants() {
                    SemanticRustcVariantsV1::Single { index } => {
                        let SemanticTypeLayoutDetailsV1::Aggregate(aggregate) = layout.details()
                        else {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                statement,
                                "single-variant enum constant has no exact Rust field layout",
                            ));
                        };
                        (*index, aggregate.field_offsets().to_vec())
                    }
                    SemanticRustcVariantsV1::Multiple(enum_layout) => {
                        let index = match enum_layout.encoding() {
                            SemanticEnumEncodingV1::Direct(direct) => {
                                let tag_size =
                                    direct.tag().primitive().size_bytes().ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "direct enum tag has an unsupported Rust width",
                                        )
                                    })?;
                                let tag_offset = base_offset
                                    .checked_add(direct.tag_offset_bytes())
                                    .ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "direct enum tag offset overflows",
                                        )
                                    })?;
                                let tag = read_constant_bits_v1(bytes, tag_offset, tag_size)
                                    .ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "direct enum tag bytes are truncated",
                                        )
                                    })?;
                                let tag_size = u8::try_from(tag_size).map_err(|_| {
                                    unsupported(
                                        0,
                                        Some(block.index()),
                                        statement,
                                        "direct enum tag exceeds the supported width",
                                    )
                                })?;
                                let tag =
                                    SemanticScalarValueV1::new(tag, tag_size).map_err(|_| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "direct enum tag has an invalid Rust representation",
                                        )
                                    })?;
                                semantic_direct_enum_variant_v1(self.types, ty, tag).ok_or_else(
                                    || {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "direct enum tag has no logical Rust variant",
                                        )
                                    },
                                )?
                            }
                            SemanticEnumEncodingV1::Niche(niche) => {
                                let tag_size =
                                    niche.tag().primitive().size_bytes().ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum tag has an unsupported Rust width",
                                        )
                                    })?;
                                let tag_offset = base_offset
                                    .checked_add(niche.source().expected_offset_bytes())
                                    .ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum tag offset overflows",
                                        )
                                    })?;
                                let tag = read_constant_bits_v1(bytes, tag_offset, tag_size)
                                    .ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum tag bytes are truncated",
                                        )
                                    })?;
                                let bits = u32::try_from(tag_size)
                                    .ok()
                                    .and_then(|size| size.checked_mul(8))
                                    .ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum tag width overflows",
                                        )
                                    })?;
                                let mask = if bits == 128 {
                                    u128::MAX
                                } else {
                                    (1_u128 << bits) - 1
                                };
                                let relative = tag.wrapping_sub(niche.niche_start()) & mask;
                                let (start, end) = niche.niche_variant_range();
                                let niche_variant_count =
                                    end.checked_sub(start).ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum variant range is reversed",
                                        )
                                    })?;
                                if relative <= u128::from(niche_variant_count) {
                                    start.checked_add(relative as u32).ok_or_else(|| {
                                        unsupported(
                                            0,
                                            Some(block.index()),
                                            statement,
                                            "niche enum variant index overflows",
                                        )
                                    })?
                                } else {
                                    niche.untagged_variant()
                                }
                            }
                        };
                        let variant_layout = enum_layout
                            .variants()
                            .iter()
                            .find(|layout| layout.variant_index() == index)
                            .ok_or_else(|| {
                                unsupported(
                                    0,
                                    Some(block.index()),
                                    statement,
                                    "enum constant has no exact Rust variant layout",
                                )
                            })?;
                        (index, variant_layout.aggregate().field_offsets().to_vec())
                    }
                    SemanticRustcVariantsV1::Empty => {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "uninhabited enum cannot be a constant",
                        ));
                    }
                };
                let logical_variant = variants.get(variant_index as usize).ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "enum constant variant is absent from its semantic type",
                    )
                })?;
                let fields = self.lower_constant_fields(
                    block,
                    statement,
                    logical_variant.fields().fields(),
                    &offsets,
                    bytes,
                    base_offset,
                    structural_nodes,
                    operations,
                )?;
                let discriminant_ty = lower_scalar_type(self.types, *discriminant)?;
                let discriminant_value = self
                    .emit(
                        operations,
                        discriminant_ty.clone(),
                        OperationKind::Constant(integer_constant(
                            &discriminant_ty,
                            logical_variant.discriminant(),
                        )?),
                    )?
                    .value()
                    .expect("emitted enum discriminant")
                    .0;
                Ok(SemanticValueBindingV1::Enum {
                    discriminant: discriminant_value,
                    discriminant_ty,
                    semantic_type: ty,
                    variant: Some(variant_index),
                    payloads: BTreeMap::from([(variant_index, fields)]),
                })
            }
            SemanticTypeShapeV1::Never
            | SemanticTypeShapeV1::Pointer(_)
            | SemanticTypeShapeV1::Slice { .. }
            | SemanticTypeShapeV1::Union(_)
            | SemanticTypeShapeV1::FunctionPointer { .. }
            | SemanticTypeShapeV1::Opaque => Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "constant Rust layout has no admitted Kernel IR value representation",
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_constant_fields(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        field_types: &[SemanticTypeIdV1],
        field_offsets: &[u64],
        bytes: &[u8],
        base_offset: u64,
        structural_nodes: &mut usize,
        operations: &mut Vec<Operation>,
    ) -> Result<Vec<SemanticValueBindingV1>, ProductionSemanticKirErrorV1> {
        if field_types.len() != field_offsets.len() {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "constant field shape disagrees with its exact Rust layout",
            ));
        }
        let mut fields = Vec::new();
        fields.try_reserve_exact(field_types.len()).map_err(|_| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "constant field count exceeds available capacity",
            )
        })?;
        for (field_type, field_offset) in field_types.iter().zip(field_offsets) {
            let offset = base_offset.checked_add(*field_offset).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    statement,
                    "constant field offset overflows",
                )
            })?;
            fields.push(self.lower_constant_bytes(
                block,
                statement,
                *field_type,
                bytes,
                offset,
                structural_nodes,
                operations,
            )?);
        }
        Ok(fields)
    }

    fn lower_terminator(
        &mut self,
        block: SemanticBlockIdV1,
        terminator: &SemanticTerminatorKindV1,
        operations: &mut Vec<Operation>,
    ) -> Result<Terminator, ProductionSemanticKirErrorV1> {
        match terminator {
            SemanticTerminatorKindV1::Goto(edge) => Ok(Terminator::Branch {
                target: BlockId(edge.target().index()),
                arguments: self.edge_arguments(block, edge.target(), operations)?,
            }),
            SemanticTerminatorKindV1::SwitchInt {
                discriminant,
                targets,
            } => {
                let (selector, selector_ty) = self
                    .lower_operand(block, None, discriminant, operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                if selector_ty == Type::BOOL {
                    let [target] = targets.values() else {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "boolean switch must have one explicit target",
                        ));
                    };
                    if target.value() > 1 {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "boolean switch target is not zero or one",
                        ));
                    }
                    let explicit = target.edge().target();
                    let otherwise = targets.otherwise().target();
                    let (then_target, else_target) = if target.value() == 1 {
                        (explicit, otherwise)
                    } else {
                        (otherwise, explicit)
                    };
                    return Ok(Terminator::ConditionalBranch {
                        condition: selector,
                        then_target: BlockId(then_target.index()),
                        then_arguments: self.edge_arguments(block, then_target, operations)?,
                        else_target: BlockId(else_target.index()),
                        else_arguments: self.edge_arguments(block, else_target, operations)?,
                    });
                }
                let cases = targets
                    .values()
                    .iter()
                    .map(|target| {
                        Ok(SwitchCase {
                            value: u64::try_from(target.value()).map_err(|_| {
                                unsupported(
                                    0,
                                    Some(block.index()),
                                    None,
                                    "switch value exceeds Kernel IR V1",
                                )
                            })?,
                            target: BlockId(target.edge().target().index()),
                            arguments: self.edge_arguments(
                                block,
                                target.edge().target(),
                                operations,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, ProductionSemanticKirErrorV1>>()?;
                Ok(Terminator::Switch {
                    selector,
                    cases,
                    default_target: BlockId(targets.otherwise().target().index()),
                    default_arguments: self.edge_arguments(
                        block,
                        targets.otherwise().target(),
                        operations,
                    )?,
                })
            }
            SemanticTerminatorKindV1::Call(call) => self.lower_call(block, call, operations),
            SemanticTerminatorKindV1::Assert {
                condition,
                expected,
                message: _,
                target,
                unwind,
            } => {
                if matches!(unwind, SemanticUnwindActionV1::Cleanup(_)) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "semantic assert has a cleanup unwind edge",
                    ));
                }
                if self.infallible_asserts.contains(&block.index()) {
                    return Ok(Terminator::Branch {
                        target: BlockId(target.target().index()),
                        arguments: self.edge_arguments(block, target.target(), operations)?,
                    });
                }
                let failure = self.assert_failure_block.ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "semantic assert has no retained runtime failure block",
                    )
                })?;
                let (condition, condition_ty) = self
                    .lower_operand(block, None, condition, operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                if condition_ty != Type::BOOL {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "semantic assert condition is not boolean",
                    ));
                }
                let success = BlockId(target.target().index());
                let success_arguments = self.edge_arguments(block, target.target(), operations)?;
                let (then_target, then_arguments, else_target, else_arguments) = if *expected {
                    (success, success_arguments, failure, vec![])
                } else {
                    (failure, vec![], success, success_arguments)
                };
                Ok(Terminator::ConditionalBranch {
                    condition,
                    then_target,
                    then_arguments,
                    else_target,
                    else_arguments,
                })
            }
            SemanticTerminatorKindV1::Abort | SemanticTerminatorKindV1::UnwindTerminate => {
                let failure = self.assert_failure_block.ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "semantic abort has no retained runtime failure block",
                    )
                })?;
                Ok(Terminator::Branch {
                    target: failure,
                    arguments: vec![],
                })
            }
            SemanticTerminatorKindV1::Return => Ok(Terminator::Return { values: vec![] }),
            SemanticTerminatorKindV1::Unreachable => Ok(Terminator::Unreachable),
            _ => Err(unsupported(
                0,
                Some(block.index()),
                None,
                unsupported_terminator_detail(terminator),
            )),
        }
    }

    fn emit_workgroup_pipeline_barrier(
        &mut self,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.push_operation(operations, || {
            Operation::new(
                Vec::new(),
                OperationKind::WorkgroupBarrier(WorkgroupBarrier {
                    memory_scope: SynchronizationScope::Workgroup,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::AcquireRelease,
                        [AddressSpace::Workgroup],
                    ),
                    convergence: Convergence::uniform(SynchronizationScope::Workgroup),
                }),
            )
        })
    }

    fn lower_workgroup_pipeline_slot(
        &mut self,
        block: SemanticBlockIdV1,
        epoch: &SemanticOperandV1,
        index: &SemanticOperandV1,
        buffers: u32,
        elements: u64,
        operations: &mut Vec<Operation>,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        let epoch = self.lower_operand(block, None, epoch, operations)?;
        let epoch = self
            .coerce_index(block, operations, epoch)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let index = self.lower_operand(block, None, index, operations)?;
        let index = self
            .coerce_index(block, operations, index)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let buffers = self.emit_index_constant(operations, u64::from(buffers))?;
        let elements = self.emit_index_constant(operations, elements)?;
        let ring_slot = self.emit_index_binary(operations, BinaryOp::Remainder, epoch, buffers)?;
        let ring_base =
            self.emit_index_binary(operations, BinaryOp::Multiply, ring_slot, elements)?;
        self.emit_index_binary(operations, BinaryOp::Add, ring_base, index)
    }

    fn pack_workgroup_pipeline_payload(
        &mut self,
        block: SemanticBlockIdV1,
        payload: SemanticValueBindingV1,
        contract: &SemanticWorkgroupPipelineTypeContractV1,
        operations: &mut Vec<Operation>,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        let components = contract
            .payload_binding
            .transport_values(&payload)
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if components.len() != contract.component_types.len()
            || components
                .iter()
                .zip(contract.component_types.iter())
                .any(|((_, actual), expected)| actual != expected)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup pipeline write payload changed physical component types",
            ));
        }
        if let [(value, ty)] = components.as_slice()
            && ty == &contract.packed_type
        {
            return Ok(*value);
        }
        let zero = pipeline_integer_constant_v1(&contract.packed_type, 0).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup pipeline packed scalar has no executable constant",
            )
        })?;
        let mut packed = self.emit_id(
            operations,
            contract.packed_type.clone(),
            OperationKind::Constant(zero),
        )?;
        let mut offset = 0_u32;
        for ((value, ty), expected) in components.iter().zip(contract.component_types.iter()) {
            debug_assert_eq!(ty, expected);
            let width = pipeline_scalar_bit_width_v1(ty).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component width is unavailable",
                )
            })?;
            let unsigned = pipeline_unsigned_component_type_v1(ty).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component has no unsigned transport",
                )
            })?;
            let mut component = *value;
            if ty != &unsigned {
                let kind = if ty == &Type::BOOL {
                    CastKind::ZeroExtend
                } else {
                    CastKind::Bitcast
                };
                component = self.emit_id(
                    operations,
                    unsigned.clone(),
                    OperationKind::Cast {
                        kind,
                        value: component,
                        to: unsigned.clone(),
                    },
                )?;
            }
            if unsigned != contract.packed_type {
                component = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Cast {
                        kind: CastKind::ZeroExtend,
                        value: component,
                        to: contract.packed_type.clone(),
                    },
                )?;
            }
            if offset != 0 {
                let shift = pipeline_integer_constant_v1(&contract.packed_type, u64::from(offset))
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline shift has no executable constant",
                        )
                    })?;
                let shift = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Constant(shift),
                )?;
                component = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Binary {
                        op: BinaryOp::ShiftLeft,
                        lhs: component,
                        rhs: shift,
                    },
                )?;
            }
            packed = self.emit_id(
                operations,
                contract.packed_type.clone(),
                OperationKind::Binary {
                    op: BinaryOp::BitOr,
                    lhs: packed,
                    rhs: component,
                },
            )?;
            offset = offset.checked_add(width).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component offset overflows",
                )
            })?;
        }
        Ok(packed)
    }

    fn unpack_workgroup_pipeline_payload(
        &mut self,
        block: SemanticBlockIdV1,
        packed: ValueId,
        contract: &SemanticWorkgroupPipelineTypeContractV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        if contract.component_types.as_ref() == [contract.packed_type.clone()] {
            return contract.payload_binding.binding_from_transport(
                self.types,
                contract.element,
                &[ValueDef::new(packed, contract.packed_type.clone())],
            );
        }
        let mut values = Vec::with_capacity(contract.component_types.len());
        let mut offset = 0_u32;
        for ty in contract.component_types.iter() {
            let width = pipeline_scalar_bit_width_v1(ty).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component width is unavailable",
                )
            })?;
            let unsigned = pipeline_unsigned_component_type_v1(ty).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component has no unsigned transport",
                )
            })?;
            let mut component = packed;
            if offset != 0 {
                let shift = pipeline_integer_constant_v1(&contract.packed_type, u64::from(offset))
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline shift has no executable constant",
                        )
                    })?;
                let shift = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Constant(shift),
                )?;
                component = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Binary {
                        op: BinaryOp::ShiftRight,
                        lhs: component,
                        rhs: shift,
                    },
                )?;
            }
            if unsigned != contract.packed_type {
                component = self.emit_id(
                    operations,
                    unsigned.clone(),
                    OperationKind::Cast {
                        kind: CastKind::Truncate,
                        value: component,
                        to: unsigned.clone(),
                    },
                )?;
            }
            if &unsigned != ty {
                component = if ty == &Type::BOOL {
                    let zero = pipeline_integer_constant_v1(&unsigned, 0).ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline Boolean transport has no zero constant",
                        )
                    })?;
                    let zero =
                        self.emit_id(operations, unsigned.clone(), OperationKind::Constant(zero))?;
                    self.emit_id(
                        operations,
                        Type::BOOL,
                        OperationKind::Compare {
                            predicate: ComparePredicate::NotEqual,
                            lhs: component,
                            rhs: zero,
                        },
                    )?
                } else {
                    self.emit_id(
                        operations,
                        ty.clone(),
                        OperationKind::Cast {
                            kind: CastKind::Bitcast,
                            value: component,
                            to: ty.clone(),
                        },
                    )?
                };
            }
            values.push(ValueDef::new(component, ty.clone()));
            offset = offset.checked_add(width).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "workgroup pipeline component offset overflows",
                )
            })?;
        }
        contract
            .payload_binding
            .binding_from_transport(self.types, contract.element, &values)
    }

    fn lower_call(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
    ) -> Result<Terminator, ProductionSemanticKirErrorV1> {
        if matches!(call.unwind(), SemanticUnwindActionV1::Cleanup(_)) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "trusted compiler intrinsic has a cleanup unwind edge",
            ));
        }
        let callable = self
            .callables
            .get(call.callee().index() as usize)
            .ok_or_else(|| unsupported(0, Some(block.index()), None, "callable is missing"))?;
        let SemanticCallableDeclV1::CompilerIntrinsic { operation, .. } = callable else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "defined and device-FFI calls require interprocedural lowering",
            ));
        };
        require_current_production_intrinsic_v1(operation)?;
        if matches!(operation, SemanticCompilerIntrinsicOperationV1::Trap) {
            self.require_call_argument_count(block, call, 0)?;
            if call.destination().is_some() {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "trap compiler intrinsic unexpectedly has a destination",
                ));
            }
            self.push_operation(operations, || {
                AmdGpuDiagnosticOperation::Trap.operation(None)
            })?;
            return Ok(Terminator::Unreachable);
        }
        let destination = call.destination().ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "compiler intrinsic call has no destination",
            )
        })?;
        let binding = match operation {
            SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent {
                dynamic_lds,
                element_storage,
                elements,
                ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let SemanticOperandV1::Move(scope_place) = &call.arguments()[0] else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS scope authority must be moved exactly once",
                    ));
                };
                if !scope_place.projections().is_empty() {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS scope authority has a projected carrier",
                    ));
                }
                let scope_local = self.require_local(block, None, scope_place.local().index())?;
                let scope = self.locals[scope_local].take().ok_or(
                    ProductionSemanticKirErrorV1::MissingLocalDefinition {
                        function: 0,
                        block: block.index(),
                        statement: None,
                        local: scope_place.local().index(),
                    },
                )?;
                if !matches!(scope, SemanticValueBindingV1::WorkgroupLdsScope) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS allocation lacks compiler-authenticated scope authority",
                    ));
                }
                let element = lower_dynamic_lds_element_type_v1(self.types, *element_storage)?;
                let storage = self
                    .types
                    .get(element_storage.index() as usize)
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "exact LDS storage type is missing",
                        )
                    })?;
                let element_size = storage.layout().size_bytes().ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS storage is dynamically sized",
                    )
                })?;
                let byte_len = elements.checked_mul(element_size).ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS byte extent overflows",
                    )
                })?;
                let extent = u32::try_from(*elements).map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exact LDS element extent exceeds Kernel IR",
                    )
                })?;
                let alignment =
                    u32::try_from(storage.layout().alignment_bytes()).map_err(|_| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "exact LDS alignment exceeds Kernel IR",
                        )
                    })?;
                let pointer_type = Type::pointer(
                    element.clone(),
                    AddressSpace::Workgroup,
                    AccessMode::ReadWrite,
                );
                let pointer = self.emit(
                    operations,
                    pointer_type,
                    OperationKind::WorkgroupMemory(WorkgroupMemory {
                        element,
                        extent: WorkgroupMemoryExtent::Static(extent),
                        alignment,
                    }),
                )?;
                let len = self.emit(
                    operations,
                    Type::INDEX,
                    OperationKind::Constant(Constant::Index(*elements)),
                )?;
                let byte_len = self.emit(
                    operations,
                    Type::INDEX,
                    OperationKind::Constant(Constant::Index(byte_len)),
                )?;
                let values = [pointer, len, byte_len]
                    .into_iter()
                    .map(|binding| {
                        let (id, ty) = binding
                            .value()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                        Ok(ValueDef::new(id, ty))
                    })
                    .collect::<Result<Vec<_>, ProductionSemanticKirErrorV1>>()?;
                binding_from_value_defs_with_validation(self.types, *dynamic_lds, &values, false)?
            }
            SemanticCompilerIntrinsicOperationV1::DynamicLdsIntoCollectiveRawParts {
                raw_parts,
                element_storage,
                element,
                ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let dynamic_lds =
                    self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::Aggregate(fields) = dynamic_lds else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "collective LDS conversion input is not the reviewed aggregate",
                    ));
                };
                if fields.len() != 6 {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "collective LDS conversion input field count changed",
                    ));
                }
                let mut pointer = fields[0].clone();
                for _ in 0..4 {
                    match pointer {
                        SemanticValueBindingV1::Aggregate(mut wrapper) if wrapper.len() == 1 => {
                            pointer = wrapper.pop().expect("singleton aggregate has one field");
                        }
                        SemanticValueBindingV1::Value { .. } => break,
                        _ => {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "collective LDS pointer wrapper changed",
                            ));
                        }
                    }
                }
                let (pointer, pointer_ty) = pointer
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                let (len, len_ty) = fields[1]
                    .clone()
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                let storage_element =
                    lower_dynamic_lds_element_type_v1(self.types, *element_storage)?;
                let semantic_element = lower_scalar_type(self.types, *element)?;
                let Type::Pointer(pointer_contract) = &pointer_ty else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "collective LDS conversion input is not a lowered pointer",
                    ));
                };
                if pointer_contract.address_space != AddressSpace::Workgroup
                    || pointer_contract.access != AccessMode::ReadWrite
                    || *pointer_contract.pointee != storage_element
                    || storage_element != semantic_element
                    || len_ty != Type::INDEX
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "collective LDS conversion pointer, element, or length changed",
                    ));
                }
                let values = [
                    ValueDef::new(pointer, pointer_ty),
                    ValueDef::new(len, len_ty),
                ];
                binding_from_value_defs_with_validation(self.types, *raw_parts, &values, false)?
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate {
                pipeline,
                buffers,
                elements,
                prefetch_distance,
                ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let scope = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(scope, SemanticValueBindingV1::WorkgroupLdsScope) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline creation lacks compiler-authenticated LDS scope",
                    ));
                }
                if !(2..=8).contains(buffers)
                    || *elements == 0
                    || *prefetch_distance == 0
                    || prefetch_distance >= buffers
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline geometry is outside the executable contract",
                    ));
                }
                let contract = self
                    .workgroup_pipeline_contracts
                    .get(pipeline)
                    .cloned()
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline has no consistent typed payload contract",
                        )
                    })?;
                let extent = u64::from(*buffers)
                    .checked_mul(*elements)
                    .and_then(|extent| u32::try_from(extent).ok())
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline LDS extent exceeds Kernel IR",
                        )
                    })?;
                let pointer_type = Type::pointer(
                    contract.packed_type.clone(),
                    AddressSpace::Workgroup,
                    AccessMode::ReadWrite,
                );
                let storage = self.emit_id(
                    operations,
                    pointer_type,
                    OperationKind::WorkgroupMemory(WorkgroupMemory {
                        element: contract.packed_type.clone(),
                        extent: WorkgroupMemoryExtent::Static(extent),
                        alignment: contract.alignment,
                    }),
                )?;
                SemanticValueBindingV1::WorkgroupPipeline {
                    storage,
                    pipeline: *pipeline,
                    element: contract.element,
                    payload_binding: contract.payload_binding,
                    component_types: contract.component_types,
                    packed_type: contract.packed_type,
                    buffers: *buffers,
                    elements: *elements,
                    prefetch_distance: *prefetch_distance,
                    alignment: contract.alignment,
                }
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineEvent { pipeline, event } => {
                self.require_call_argument_count(block, call, 2)?;
                let receiver = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::WorkgroupPipeline {
                    pipeline: actual_pipeline,
                    buffers,
                    elements,
                    prefetch_distance,
                    ..
                } = receiver
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline event lacks its compiler-owned storage capability",
                    ));
                };
                if actual_pipeline != *pipeline
                    || !(2..=8).contains(&buffers)
                    || elements == 0
                    || prefetch_distance == 0
                    || prefetch_distance >= buffers
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline event changed its authenticated contract",
                    ));
                }
                let epoch = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                let _ = self.coerce_index(block, operations, epoch)?;
                if *event == SemanticWorkgroupPipelineEventV1::Wait {
                    self.emit_workgroup_pipeline_barrier(operations)?;
                }
                SemanticValueBindingV1::Unit
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite { pipeline, element } => {
                self.require_call_argument_count(block, call, 4)?;
                let receiver = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::WorkgroupPipeline {
                    storage,
                    pipeline: actual_pipeline,
                    element: actual_element,
                    payload_binding,
                    component_types,
                    packed_type,
                    buffers,
                    elements,
                    prefetch_distance: _,
                    alignment,
                } = receiver
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline write lacks its compiler-owned storage capability",
                    ));
                };
                let contract = self
                    .workgroup_pipeline_contracts
                    .get(pipeline)
                    .cloned()
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline write has no typed payload contract",
                        )
                    })?;
                if actual_pipeline != *pipeline
                    || actual_element != *element
                    || contract.element != *element
                    || payload_binding != contract.payload_binding
                    || component_types.as_ref() != contract.component_types.as_ref()
                    || packed_type != contract.packed_type
                    || alignment != contract.alignment
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline write changed its authenticated storage contract",
                    ));
                }
                let slot = self.lower_workgroup_pipeline_slot(
                    block,
                    &call.arguments()[1],
                    &call.arguments()[2],
                    buffers,
                    elements,
                    operations,
                )?;
                let payload = self.lower_operand(block, None, &call.arguments()[3], operations)?;
                let packed =
                    self.pack_workgroup_pipeline_payload(block, payload, &contract, operations)?;
                let pointer = self.emit_id(
                    operations,
                    Type::pointer(
                        contract.packed_type.clone(),
                        AddressSpace::Workgroup,
                        AccessMode::ReadWrite,
                    ),
                    OperationKind::GetElementPointer {
                        base: storage,
                        offset: slot,
                    },
                )?;
                self.push_operation(operations, || {
                    Operation::new(
                        Vec::new(),
                        OperationKind::Store {
                            pointer,
                            value: packed,
                            access: MemoryAccess::new(AddressSpace::Workgroup, alignment),
                        },
                    )
                })?;
                SemanticValueBindingV1::Unit
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineRead { pipeline, element } => {
                self.require_call_argument_count(block, call, 3)?;
                let receiver = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::WorkgroupPipeline {
                    storage,
                    pipeline: actual_pipeline,
                    element: actual_element,
                    payload_binding,
                    component_types,
                    packed_type,
                    buffers,
                    elements,
                    prefetch_distance: _,
                    alignment,
                } = receiver
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline read lacks its compiler-owned storage capability",
                    ));
                };
                let contract = self
                    .workgroup_pipeline_contracts
                    .get(pipeline)
                    .cloned()
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "workgroup pipeline read has no typed payload contract",
                        )
                    })?;
                if actual_pipeline != *pipeline
                    || actual_element != *element
                    || contract.element != *element
                    || payload_binding != contract.payload_binding
                    || component_types.as_ref() != contract.component_types.as_ref()
                    || packed_type != contract.packed_type
                    || alignment != contract.alignment
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "workgroup pipeline read changed its authenticated storage contract",
                    ));
                }
                let slot = self.lower_workgroup_pipeline_slot(
                    block,
                    &call.arguments()[1],
                    &call.arguments()[2],
                    buffers,
                    elements,
                    operations,
                )?;
                let pointer = self.emit_id(
                    operations,
                    Type::pointer(
                        contract.packed_type.clone(),
                        AddressSpace::Workgroup,
                        AccessMode::ReadWrite,
                    ),
                    OperationKind::GetElementPointer {
                        base: storage,
                        offset: slot,
                    },
                )?;
                let packed = self.emit_id(
                    operations,
                    contract.packed_type.clone(),
                    OperationKind::Load {
                        pointer,
                        access: MemoryAccess::new(AddressSpace::Workgroup, alignment),
                    },
                )?;
                self.unpack_workgroup_pipeline_payload(block, packed, &contract, operations)?
            }
            SemanticCompilerIntrinsicOperationV1::MathContextCurrent { .. } => {
                self.require_call_argument_count(block, call, 0)?;
                SemanticValueBindingV1::MathContext
            }
            SemanticCompilerIntrinsicOperationV1::MathF32 { function, .. } => {
                self.require_call_argument_count(block, call, function.arity() + 1)?;
                let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(context, SemanticValueBindingV1::MathContext) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "device math operation lacks compiler-issued math authority",
                    ));
                }
                let function = lower_f32_math_function(*function);
                let mut arguments = Vec::with_capacity(function.arity());
                for argument in &call.arguments()[1..] {
                    let (id, ty) = self
                        .lower_operand(block, None, argument, operations)?
                        .value()
                        .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                    if ty != Type::Scalar(ScalarType::F32) {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "device math argument is not f32",
                        ));
                    }
                    arguments.push(id);
                }
                self.emit_float_operation(
                    operations,
                    FloatOperation::F32Math {
                        function,
                        implementation: function.required_implementation(),
                        arguments,
                    },
                )?
            }
            SemanticCompilerIntrinsicOperationV1::Bf16Conversion {
                kind,
                input,
                output,
            } => {
                self.require_call_argument_count(block, call, 1)?;
                if semantic_operand_type(&call.arguments()[0]) != *input
                    || destination.place().ty() != *output
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "BF16 conversion semantic input or output type changed",
                    ));
                }
                let argument = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                match kind {
                    SemanticBf16ConversionKindV1::FromBits => {
                        let (bits, ty) = argument
                            .value()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                        if ty != Type::Scalar(ScalarType::U16) {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 from_bits input is not u16",
                            ));
                        }
                        binding_from_value_defs(self.types, *output, &[ValueDef::new(bits, ty)])?
                    }
                    SemanticBf16ConversionKindV1::ToBits => {
                        let values = argument
                            .values()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                        let [(bits, ty)] = values.as_slice() else {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 to_bits storage is not one scalar",
                            ));
                        };
                        if *ty != Type::Scalar(ScalarType::U16) {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 to_bits storage is not u16",
                            ));
                        }
                        binding_from_value_defs(
                            self.types,
                            *output,
                            &[ValueDef::new(*bits, ty.clone())],
                        )?
                    }
                    SemanticBf16ConversionKindV1::FromF32RoundTiesEven => {
                        let (value, ty) = argument
                            .value()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                        if ty != Type::Scalar(ScalarType::F32) {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 from_f32 input is not f32",
                            ));
                        }
                        let (narrowed, narrowed_ty) = self
                            .emit_float_operation(
                                operations,
                                FloatOperation::Convert {
                                    kind: FloatConversionKind::F32ToBf16RoundTiesEven,
                                    value,
                                },
                            )?
                            .value()
                            .expect("BF16 conversion emits one value");
                        let bits_ty = Type::Scalar(ScalarType::U16);
                        let bits = self.emit_id(
                            operations,
                            bits_ty.clone(),
                            OperationKind::Cast {
                                kind: CastKind::Bitcast,
                                value: narrowed,
                                to: bits_ty.clone(),
                            },
                        )?;
                        debug_assert_eq!(narrowed_ty, Type::Scalar(ScalarType::Bf16));
                        binding_from_value_defs(
                            self.types,
                            *output,
                            &[ValueDef::new(bits, bits_ty)],
                        )?
                    }
                    SemanticBf16ConversionKindV1::ToF32 => {
                        let values = argument
                            .values()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                        let [(bits, ty)] = values.as_slice() else {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 to_f32 storage is not one scalar",
                            ));
                        };
                        if *ty != Type::Scalar(ScalarType::U16) {
                            return Err(unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "BF16 to_f32 storage is not u16",
                            ));
                        }
                        let bf16 = self.emit_id(
                            operations,
                            Type::Scalar(ScalarType::Bf16),
                            OperationKind::Cast {
                                kind: CastKind::Bitcast,
                                value: *bits,
                                to: Type::Scalar(ScalarType::Bf16),
                            },
                        )?;
                        self.emit_float_operation(
                            operations,
                            FloatOperation::Convert {
                                kind: FloatConversionKind::Bf16ToF32,
                                value: bf16,
                            },
                        )?
                    }
                }
            }
            SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { .. } => {
                self.require_call_argument_count(block, call, 0)?;
                SemanticValueBindingV1::CollectiveContext
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupReduceSum { element, .. } => {
                self.lower_workgroup_reduce_sum(block, call, operations, *element)?
            }
            SemanticCompilerIntrinsicOperationV1::SubgroupReduceF32 { width, kind, .. } => {
                self.require_call_argument_count(block, call, 2)?;
                let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "subgroup reduction lacks compiler-issued collective authority",
                    ));
                }
                let value = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                self.lower_subgroup_reduce_f32(block, operations, value, *width, *kind)?
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950SubgroupContextCurrent { .. } => {
                self.require_call_argument_count(block, call, 0)?;
                SemanticValueBindingV1::CollectiveContext
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950SubgroupReduceF32 {
                width, kind, ..
            } => {
                self.require_call_argument_count(block, call, 2)?;
                let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "gfx950 subgroup reduction lacks compiler-issued authority",
                    ));
                }
                let (value, ty) = self
                    .lower_operand(block, None, &call.arguments()[1], operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                if ty != Type::Scalar(ScalarType::F32)
                    || *width == 0
                    || !width.is_power_of_two()
                    || *width > 64
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "gfx950 subgroup reduction type or width changed",
                    ));
                }
                let kind = match kind {
                    SemanticSubgroupReductionKindV1::Sum => WaveF32ReductionKindV1::Sum,
                    SemanticSubgroupReductionKindV1::Maximum => WaveF32ReductionKindV1::Maximum,
                };
                self.emit(
                    operations,
                    Type::Scalar(ScalarType::F32),
                    OperationKind::Wave(WaveOperation::full(
                        WaveOperationKind::ReduceF32 {
                            value,
                            tile_width: *width,
                            kind,
                        },
                        WaveWidth::Wave64,
                    )),
                )?
            }
            SemanticCompilerIntrinsicOperationV1::SubgroupBroadcastF32 { width, .. } => {
                self.require_call_argument_count(block, call, 3)?;
                let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "gfx950 subgroup broadcast lacks compiler-issued authority",
                    ));
                }
                let (value, value_ty) = self
                    .lower_operand(block, None, &call.arguments()[1], operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                let authenticated_source_bound = authenticated_unsigned_operand_exclusive_bound_v1(
                    self.types,
                    self.function,
                    &self.authenticated_loop_induction_bounds,
                    block,
                    &call.arguments()[2],
                );
                let (source_lane, source_ty) = self
                    .lower_operand(block, None, &call.arguments()[2], operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                let authenticated_source_bound = authenticated_source_bound.filter(|bound| {
                    source_ty == Type::Scalar(ScalarType::U32)
                        && authenticated_subgroup_broadcast_source_is_bounded(*bound, *width)
                });
                let source_lane = if authenticated_source_bound.is_some() {
                    let mask = self
                        .emit(
                            operations,
                            Type::Scalar(ScalarType::U32),
                            OperationKind::Constant(Constant::U32(*width - 1)),
                        )?
                        .value()
                        .expect("authenticated subgroup mask has one value")
                        .0;
                    self.emit(
                        operations,
                        Type::Scalar(ScalarType::U32),
                        OperationKind::Binary {
                            op: BinaryOp::BitAnd,
                            lhs: source_lane,
                            rhs: mask,
                        },
                    )?
                    .value()
                    .expect("authenticated subgroup source mask has one value")
                    .0
                } else {
                    source_lane
                };
                if let Some(bound) = authenticated_source_bound {
                    self.emitted_unsigned_exclusive_bounds
                        .insert(source_lane, bound);
                }
                let bounded_source = subgroup_broadcast_source_is_statically_bounded(
                    operations,
                    source_lane,
                    *width,
                ) || self
                    .emitted_u32_constants
                    .get(&source_lane)
                    .is_some_and(|lane| *lane < *width)
                    || self
                        .emitted_u32_bitand_masks
                        .get(&source_lane)
                        .is_some_and(|mask| *mask < *width)
                    || self
                        .emitted_unsigned_exclusive_bounds
                        .get(&source_lane)
                        .is_some_and(|bound| {
                            authenticated_subgroup_broadcast_source_is_bounded(*bound, *width)
                        });
                if value_ty != Type::Scalar(ScalarType::F32)
                    || source_ty != Type::Scalar(ScalarType::U32)
                    || *width == 0
                    || !width.is_power_of_two()
                    || *width > 64
                    || !bounded_source
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "gfx950 subgroup broadcast requires f32, a valid width, and a statically bounded source lane",
                    ));
                }
                self.emit(
                    operations,
                    Type::Scalar(ScalarType::F32),
                    OperationKind::Wave(WaveOperation::full(
                        WaveOperationKind::BroadcastF32 {
                            value,
                            source_lane,
                            tile_width: *width,
                        },
                        WaveWidth::Wave64,
                    )),
                )?
            }
            SemanticCompilerIntrinsicOperationV1::MatrixContextCurrent { .. } => {
                self.require_call_argument_count(block, call, 0)?;
                SemanticValueBindingV1::MatrixContext
            }
            SemanticCompilerIntrinsicOperationV1::WaveLaneCurrent { lane, wave_width } => {
                self.require_call_argument_count(block, call, 0)?;
                if *wave_width != 64 {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "typed MFMA lane requires the authenticated wave64 profile",
                    ));
                }
                let lane_id = self.emit_results(
                    operations,
                    vec![Type::Scalar(ScalarType::U32)],
                    OperationKind::Wave(WaveOperation::full(
                        WaveOperationKind::LaneId,
                        WaveWidth::Wave64,
                    )),
                )?;
                let lane_binding = binding_from_value_defs(self.types, *lane, &lane_id)?;
                let value = require_single_u32_component(
                    block,
                    lane_binding,
                    "typed MFMA lane has no exact u32 representation",
                )?;
                SemanticValueBindingV1::WaveLane {
                    value,
                    wave: SemanticCurrentWaveV1::new(*wave_width),
                }
            }
            SemanticCompilerIntrinsicOperationV1::Bf16MatrixViewRowMajor {
                result,
                view,
                error,
                ..
            } => self.lower_checked_strided_read_view(
                block,
                call,
                operations,
                *result,
                *view,
                *error,
                Type::Scalar(ScalarType::U16),
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950Fp4MatrixViewRowMajor {
                result,
                view,
                error,
                ..
            }
            | SemanticCompilerIntrinsicOperationV1::Gfx950Fp8MatrixViewRowMajor {
                result,
                view,
                error,
                ..
            } => self.lower_checked_strided_read_view(
                block,
                call,
                operations,
                *result,
                *view,
                *error,
                Type::Scalar(ScalarType::U8),
            )?,
            SemanticCompilerIntrinsicOperationV1::StridedReadView2DFromSharedSlice {
                result,
                view,
                error,
                element,
            } => self.lower_checked_strided_read_view(
                block,
                call,
                operations,
                *result,
                *view,
                *error,
                lower_scalar_type(self.types, *element)?,
            )?,
            SemanticCompilerIntrinsicOperationV1::StridedReadView2DLoadOr { element, .. } => self
                .lower_strided_read_view_load_or(
                block,
                call,
                operations,
                lower_scalar_type(self.types, *element)?,
            )?,
            SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoad { .. } => {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "the retired Option-returning BF16 matrix load is not admitted; use Bf16MatrixLoadZeroFilledV2",
                ));
            }
            SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoadZeroFilledV2 {
                contract,
                storage_layout,
                ..
            } => {
                self.lower_bf16_matrix_load(block, call, operations, *contract, *storage_layout)?
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950Fp8MatrixLoadM16K128 {
                contract,
                storage_layout,
                ..
            } => self.lower_gfx950_low_precision_matrix_load(
                block,
                call,
                operations,
                *contract,
                *storage_layout,
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950Fp4MatrixLoadM16K128 {
                contract,
                storage_layout,
                ..
            } => self.lower_gfx950_low_precision_matrix_load(
                block,
                call,
                operations,
                *contract,
                *storage_layout,
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeCurrent { format, .. } => {
                self.lower_gfx950_lds_transpose_current(block, call, operations, *format)?
            }
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeStage {
                input_tile,
                format,
                ..
            } => self.lower_gfx950_lds_transpose_stage(
                block,
                call,
                operations,
                *input_tile,
                *format,
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposePublish {
                input_tile,
                format,
                ..
            } => self.lower_gfx950_lds_transpose_publish(
                block,
                call,
                operations,
                *input_tile,
                *format,
            )?,
            SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeRead {
                tile,
                contract,
                format,
                ..
            } => self.lower_gfx950_lds_transpose_read(
                block, call, operations, *tile, *contract, *format,
            )?,
            SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorZero {
                fragment,
                contract,
                ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let (_, wave) = require_current_wave_lane(
                    block,
                    self.lower_operand(block, None, &call.arguments()[0], operations)?,
                    contract.wave_width,
                    "zero accumulator lane",
                )?;
                let mut values = Vec::with_capacity(4);
                for _ in 0..4 {
                    let (id, ty) = self
                        .emit(
                            operations,
                            Type::Scalar(ScalarType::F32),
                            OperationKind::Constant(Constant::F32Bits(0.0_f32.to_bits())),
                        )?
                        .value()
                        .expect("emitted zero accumulator component");
                    values.push((id, ty));
                }
                let _ = fragment;
                SemanticValueBindingV1::AccumulatorFragment {
                    values,
                    contract: *contract,
                    wave,
                }
            }
            SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorIntoValues {
                values, ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let fragment = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::AccumulatorFragment {
                    values: fragment, ..
                } = fragment
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "FP32 matrix accumulator lacks typed producer metadata",
                    ));
                };
                let fragment = require_components(
                    block,
                    fragment,
                    Type::Scalar(ScalarType::F32),
                    4,
                    "FP32 matrix accumulator fragment",
                )?
                .into_iter()
                .map(|(id, ty)| ValueDef::new(id, ty))
                .collect::<Vec<_>>();
                binding_from_value_defs(self.types, *values, &fragment)?
            }
            SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                accumulator_fragment,
                lhs: expected_lhs,
                rhs: expected_rhs,
                accumulator: expected_accumulator,
                ..
            } => {
                self.require_call_argument_count(block, call, 4)?;
                let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                if !matches!(context, SemanticValueBindingV1::MatrixContext) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "matrix operation lacks compiler-issued context authority",
                    ));
                }
                let lhs = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                let rhs = self.lower_operand(block, None, &call.arguments()[2], operations)?;
                let accumulator =
                    self.lower_operand(block, None, &call.arguments()[3], operations)?;
                let SemanticValueBindingV1::MatrixFragment {
                    values: lhs,
                    contract: lhs_contract,
                    storage_layout: lhs_storage,
                    wave: lhs_wave,
                } = lhs
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "matrix lhs lacks an authenticated checked-load producer",
                    ));
                };
                let SemanticValueBindingV1::MatrixFragment {
                    values: rhs,
                    contract: rhs_contract,
                    storage_layout: rhs_storage,
                    wave: rhs_wave,
                } = rhs
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "matrix rhs lacks an authenticated checked-load producer",
                    ));
                };
                let SemanticValueBindingV1::AccumulatorFragment {
                    values: accumulator,
                    contract: accumulator_contract,
                    wave: accumulator_wave,
                } = accumulator
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "matrix accumulator lacks an authenticated zero/MFMA producer",
                    ));
                };
                if lhs_contract != *expected_lhs
                    || rhs_contract != *expected_rhs
                    || accumulator_contract != *expected_accumulator
                    || lhs_wave != rhs_wave
                    || lhs_wave != accumulator_wave
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "matrix operand producer contracts or wave associations differ",
                    ));
                }
                if matches!(
                    expected_accumulator.profile,
                    SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                        | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
                ) {
                    if lhs_storage != SemanticMfmaStorageLayoutV1::RowMajor
                        || rhs_storage != SemanticMfmaStorageLayoutV1::RowMajor
                    {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "gfx950 low-precision matrix operands require checked row-major producers",
                        ));
                    }
                    let lhs = require_components(
                        block,
                        lhs,
                        Type::Scalar(ScalarType::U32),
                        8,
                        "gfx950 low-precision matrix lhs fragment",
                    )?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("eight checked gfx950 lhs dwords");
                    let rhs = require_components(
                        block,
                        rhs,
                        Type::Scalar(ScalarType::U32),
                        8,
                        "gfx950 low-precision matrix rhs fragment",
                    )?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("eight checked gfx950 rhs dwords");
                    let accumulator = require_components(
                        block,
                        accumulator,
                        Type::Scalar(ScalarType::F32),
                        4,
                        "gfx950 low-precision matrix accumulator fragment",
                    )?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("four checked gfx950 accumulator components");
                    let matrix = if expected_accumulator.profile
                        == SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                    {
                        let layout = if expected_lhs.profile
                            == SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                            && expected_rhs.profile == SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
                        {
                            TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_fp8_e4m3_f32_m16n16k128_wave64()
                        } else {
                            TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_f32_m16n16k128_wave64()
                        };
                        MatrixOperation::scaled_multiply_accumulate_fp4_e2m1(lhs, rhs, accumulator)
                            .with_declared_tensor_layout(layout)
                    } else {
                        MatrixOperation::scaled_multiply_accumulate_fp8_e4m3(
                            lhs,
                            rhs,
                            accumulator,
                        )
                        .with_declared_tensor_layout(
                            TensorLayoutContractV1::gfx950_scaled_mfma_fp8_e4m3_f32_m16n16k128_wave64(),
                        )
                    };
                    let results = self.emit_results(
                        operations,
                        vec![Type::Scalar(ScalarType::F32); 4],
                        OperationKind::Matrix(matrix),
                    )?;
                    let _ = accumulator_fragment;
                    SemanticValueBindingV1::AccumulatorFragment {
                        values: results
                            .into_iter()
                            .map(|value| (value.id, value.ty))
                            .collect(),
                        contract: accumulator_contract,
                        wave: accumulator_wave,
                    }
                } else {
                    let lhs = require_components(
                        block,
                        lhs,
                        Type::Scalar(ScalarType::Bf16),
                        4,
                        "matrix lhs fragment",
                    )?;
                    let rhs = require_components(
                        block,
                        rhs,
                        Type::Scalar(ScalarType::Bf16),
                        4,
                        "matrix rhs fragment",
                    )?;
                    let accumulator = require_components(
                        block,
                        accumulator,
                        Type::Scalar(ScalarType::F32),
                        4,
                        "matrix accumulator fragment",
                    )?;
                    let lhs = lhs
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("four checked lhs components");
                    let rhs = rhs
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("four checked rhs components");
                    let accumulator = accumulator
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("four checked accumulator components");
                    let mut tensor_layout =
                        TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64()
                            .with_zero_filled_predicate_inputs();
                    if lhs_storage == SemanticMfmaStorageLayoutV1::LdsXor4 {
                        tensor_layout = tensor_layout.with_a_lds_xor4();
                    }
                    if rhs_storage == SemanticMfmaStorageLayoutV1::LdsXor4 {
                        tensor_layout = tensor_layout.with_b_lds_xor4();
                    }
                    let results = self.emit_results(
                        operations,
                        vec![Type::Scalar(ScalarType::F32); 4],
                        OperationKind::Matrix(
                            MatrixOperation::multiply_accumulate(lhs, rhs, accumulator)
                                .with_declared_tensor_layout(tensor_layout),
                        ),
                    )?;
                    let _ = accumulator_fragment;
                    SemanticValueBindingV1::AccumulatorFragment {
                        values: results
                            .into_iter()
                            .map(|value| (value.id, value.ty))
                            .collect(),
                        contract: accumulator_contract,
                        wave: accumulator_wave,
                    }
                }
            }
            SemanticCompilerIntrinsicOperationV1::ThreadIndex1d { .. } => {
                if !call.arguments().is_empty() {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "thread index intrinsic has arguments",
                    ));
                }
                let (id, _) = self
                    .emit(
                        operations,
                        Type::INDEX,
                        OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
                    )?
                    .value()
                    .expect("emitted index value");
                SemanticValueBindingV1::IndexWitness {
                    id,
                    index_space: SemanticDisjointIndexSpaceV1::Index1d,
                    disjoint: false,
                    availability: None,
                }
            }
            SemanticCompilerIntrinsicOperationV1::ThreadIndexGet { .. } => {
                self.require_call_argument_count(block, call, 1)?;
                self.lower_operand(block, None, &call.arguments()[0], operations)?
            }
            SemanticCompilerIntrinsicOperationV1::ThreadIndexIntoDisjoint {
                index_space, ..
            } => {
                self.require_call_argument_count(block, call, 1)?;
                let binding = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::IndexWitness {
                    id,
                    availability,
                    index_space: actual,
                    disjoint: false,
                } = binding
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "into_disjoint receiver is not a thread-index witness",
                    ));
                };
                if actual != *index_space {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "into_disjoint mapping identity changed",
                    ));
                }
                SemanticValueBindingV1::IndexWitness {
                    availability,
                    id,
                    index_space: actual,
                    disjoint: true,
                }
            }
            SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedShift {
                input_space,
                output_space,
                offset,
                ..
            } => self.lower_checked_shift(
                block,
                call,
                operations,
                *input_space,
                *output_space,
                *offset,
                false,
            )?,
            SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedBlock {
                input_space,
                output_space,
                lanes_per_block,
                elements_per_lane,
                ..
            } => self.lower_checked_block(
                block,
                call,
                operations,
                *input_space,
                *output_space,
                *lanes_per_block,
                *elements_per_lane,
            )?,
            SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedTiled2d {
                input_space,
                output_space,
                lanes_per_tile,
                tile_rows,
                tile_columns,
                elements_per_lane,
                ..
            } => self.lower_checked_tiled_2d(
                block,
                call,
                operations,
                *input_space,
                *output_space,
                *lanes_per_tile,
                *tile_rows,
                *tile_columns,
                *elements_per_lane,
            )?,
            SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedRowStriped2d {
                input_space,
                output_space,
                lanes_per_row,
                elements_per_lane,
                ..
            } => self.lower_checked_row_striped_2d(
                block,
                call,
                operations,
                *input_space,
                *output_space,
                *lanes_per_row,
                *elements_per_lane,
            )?,
            SemanticCompilerIntrinsicOperationV1::DisjointIndexGet { index_space, .. } => {
                self.require_call_argument_count(block, call, 1)?;
                let binding = self.lower_operand(block, None, &call.arguments()[0], operations)?;
                let SemanticValueBindingV1::IndexWitness {
                    id,
                    index_space: actual,
                    disjoint: true,
                    ..
                } = binding
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "DisjointIndex::get receiver is not disjoint authority",
                    ));
                };
                if actual != *index_space {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "DisjointIndex::get mapping identity changed",
                    ));
                }
                SemanticValueBindingV1::Value {
                    id,
                    ty: Type::INDEX,
                }
            }
            SemanticCompilerIntrinsicOperationV1::DisjointIndexCheckedShift {
                input_space,
                output_space,
                offset,
                ..
            } => self.lower_checked_shift(
                block,
                call,
                operations,
                *input_space,
                *output_space,
                *offset,
                true,
            )?,
            SemanticCompilerIntrinsicOperationV1::ThreadIndex(axis) => {
                self.emit_launch_index_v1(operations, IndexKind::Local, lower_axis(*axis))?
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupIndex(axis) => {
                self.emit_launch_index_v1(operations, IndexKind::Workgroup, lower_axis(*axis))?
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupDimension(axis) => {
                self.emit_launch_index_v1(operations, IndexKind::WorkgroupSize, lower_axis(*axis))?
            }
            SemanticCompilerIntrinsicOperationV1::GridDimension(axis) => {
                self.emit_launch_index_v1(operations, IndexKind::WorkgroupCount, lower_axis(*axis))?
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceLen { .. } => {
                self.require_call_argument_count(block, call, 1)?;
                let (slice, slice_ty) = self
                    .lower_operand(block, None, &call.arguments()[0], operations)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                if !matches!(slice_ty, Type::Slice(_)) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "DisjointSlice::len receiver is not a lowered slice",
                    ));
                }
                let (id, _) = self
                    .emit(
                        operations,
                        Type::INDEX,
                        OperationKind::SliceLength { slice },
                    )?
                    .value()
                    .expect("emitted slice length");
                SemanticValueBindingV1::Value {
                    id,
                    ty: Type::INDEX,
                }
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut { .. } => {
                self.require_call_argument_count(block, call, 2)?;
                let index_binding =
                    self.lower_operand(block, None, &call.arguments()[1], operations)?;
                if !matches!(
                    index_binding,
                    SemanticValueBindingV1::IndexWitness {
                        index_space: SemanticDisjointIndexSpaceV1::Index1d,
                        disjoint: false,
                        ..
                    }
                ) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "DisjointSlice::get_mut requires the identity thread-index witness",
                    ));
                }
                self.lower_checked_slice_access(block, call, operations, 0, index_binding, None)?
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetDisjointMut {
                index_space,
                ..
            } => {
                self.require_call_argument_count(block, call, 2)?;
                let index_binding =
                    self.lower_operand(block, None, &call.arguments()[1], operations)?;
                if !matches!(index_binding, SemanticValueBindingV1::IndexWitness {
                    index_space: actual,
                    disjoint: true,
                    ..
                } if actual == *index_space)
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_disjoint_mut mapping authority does not match the slice",
                    ));
                }
                self.lower_checked_slice_access(block, call, operations, 0, index_binding, None)?
            }
            SemanticCompilerIntrinsicOperationV1::GridLeaderCurrent { .. } => {
                self.require_call_argument_count(block, call, 0)?;
                let (index, _) = self
                    .emit(
                        operations,
                        Type::INDEX,
                        OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
                    )?
                    .value()
                    .expect("emitted index value");
                let (one, _) = self
                    .emit(
                        operations,
                        Type::INDEX,
                        OperationKind::Constant(Constant::Index(1)),
                    )?
                    .value()
                    .expect("emitted index constant");
                let (present, _) = self
                    .emit(
                        operations,
                        Type::BOOL,
                        OperationKind::Compare {
                            predicate: ComparePredicate::LessThan,
                            lhs: index,
                            rhs: one,
                        },
                    )?
                    .value()
                    .expect("emitted leader predicate");
                let availability = self
                    .option_dominance
                    .availability(destination.place().local())
                    .ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block.index()),
                            None,
                            "grid-leader Option lacks an authenticated Some edge",
                        )
                    })?;
                SemanticValueBindingV1::OptionGridLeader {
                    present,
                    availability,
                }
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMutExclusive { .. } => {
                self.require_call_argument_count(block, call, 3)?;
                let leader = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                if !matches!(leader, SemanticValueBindingV1::GridLeader { .. }) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "exclusive access lacks grid-leader authority",
                    ));
                }
                let index = self.lower_operand(block, None, &call.arguments()[2], operations)?;
                let index = self.coerce_index(block, operations, index)?;
                self.lower_checked_slice_access(block, call, operations, 0, index, None)?
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetBlockMut {
                index_space,
                lanes_per_block,
                elements_per_lane,
                ..
            } => {
                self.require_call_argument_count(block, call, 3)?;
                let witness = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                let SemanticValueBindingV1::ComponentWitness {
                    raw,
                    index_space: actual,
                    ..
                } = witness
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_block_mut lacks blocked ownership authority",
                    ));
                };
                let expected = SemanticDisjointIndexSpaceV1::BlockedIndex1d {
                    lanes_per_block: *lanes_per_block,
                    elements_per_lane: *elements_per_lane,
                };
                if actual != expected || *index_space != expected {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_block_mut mapping identity changed",
                    ));
                }
                let component =
                    self.lower_operand(block, None, &call.arguments()[2], operations)?;
                let component = self.coerce_index(block, operations, component)?;
                let (component, _) = component
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
                let (index, present) = self.lower_block_component_index(
                    block,
                    operations,
                    raw,
                    component,
                    *lanes_per_block,
                    *elements_per_lane,
                )?;
                self.lower_checked_slice_access(
                    block,
                    call,
                    operations,
                    0,
                    SemanticValueBindingV1::Value {
                        id: index,
                        ty: Type::INDEX,
                    },
                    Some(present),
                )?
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetTiled2dMut {
                index_space,
                lanes_per_tile,
                tile_rows,
                tile_columns,
                elements_per_lane,
                ..
            } => {
                self.require_call_argument_count(block, call, 6)?;
                let witness = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                let SemanticValueBindingV1::ComponentWitness {
                    raw,
                    index_space: actual,
                    ..
                } = witness
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_tiled_2d_mut lacks tiled ownership authority",
                    ));
                };
                let expected = SemanticDisjointIndexSpaceV1::Tiled2dIndex1d {
                    lanes_per_tile: *lanes_per_tile,
                    tile_rows: *tile_rows,
                    tile_columns: *tile_columns,
                    elements_per_lane: *elements_per_lane,
                };
                if actual != expected || *index_space != expected {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_tiled_2d_mut mapping identity changed",
                    ));
                }
                let mut indices = Vec::with_capacity(4);
                for argument in &call.arguments()[2..6] {
                    let value = self.lower_operand(block, None, argument, operations)?;
                    let value = self.coerce_index(block, operations, value)?;
                    indices.push(
                        value
                            .value()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
                            .0,
                    );
                }
                let [component, rows, columns, row_stride] = indices
                    .try_into()
                    .expect("four checked tiled-2d index operands");
                let (index, present) = self.lower_tiled_2d_component_index(
                    block,
                    operations,
                    raw,
                    component,
                    rows,
                    columns,
                    row_stride,
                    *lanes_per_tile,
                    *tile_rows,
                    *tile_columns,
                    *elements_per_lane,
                )?;
                self.lower_checked_slice_access(
                    block,
                    call,
                    operations,
                    0,
                    SemanticValueBindingV1::Value {
                        id: index,
                        ty: Type::INDEX,
                    },
                    Some(present),
                )?
            }
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetRowStriped2dMut {
                index_space,
                lanes_per_row,
                elements_per_lane,
                ..
            } => {
                self.require_call_argument_count(block, call, 6)?;
                let witness = self.lower_operand(block, None, &call.arguments()[1], operations)?;
                let SemanticValueBindingV1::ComponentWitness {
                    raw,
                    index_space: actual,
                    ..
                } = witness
                else {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_row_striped_2d_mut lacks row ownership authority",
                    ));
                };
                let expected = SemanticDisjointIndexSpaceV1::RowStriped2dIndex1d {
                    lanes_per_row: *lanes_per_row,
                    elements_per_lane: *elements_per_lane,
                };
                if actual != expected || *index_space != expected {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "get_row_striped_2d_mut mapping identity changed",
                    ));
                }
                let mut indices = Vec::with_capacity(4);
                for argument in &call.arguments()[2..6] {
                    let value = self.lower_operand(block, None, argument, operations)?;
                    let value = self.coerce_index(block, operations, value)?;
                    indices.push(
                        value
                            .value()
                            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
                            .0,
                    );
                }
                let [component, rows, columns, row_stride] = indices.try_into().map_err(|_| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "row-striped operand count changed",
                    )
                })?;
                let (index, present) = self.lower_row_striped_2d_component_index(
                    block,
                    operations,
                    raw,
                    component,
                    rows,
                    columns,
                    row_stride,
                    *lanes_per_row,
                    *elements_per_lane,
                )?;
                self.lower_checked_slice_access(
                    block,
                    call,
                    operations,
                    0,
                    SemanticValueBindingV1::Value {
                        id: index,
                        ty: Type::INDEX,
                    },
                    Some(present),
                )?
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupBarrier => {
                self.require_call_argument_count(block, call, 0)?;
                self.push_operation(operations, || {
                    Operation::new(
                        Vec::new(),
                        OperationKind::WorkgroupBarrier(WorkgroupBarrier {
                            memory_scope: SynchronizationScope::Workgroup,
                            semantics: BarrierSemantics::new(
                                MemoryOrdering::AcquireRelease,
                                [AddressSpace::Workgroup],
                            ),
                            convergence: Convergence::uniform(SynchronizationScope::Workgroup),
                        }),
                    )
                })?;
                SemanticValueBindingV1::Unit
            }
            SemanticCompilerIntrinsicOperationV1::ColdPath => {
                self.require_call_argument_count(block, call, 0)?;
                SemanticValueBindingV1::Unit
            }
            SemanticCompilerIntrinsicOperationV1::WaveBarrier
            | SemanticCompilerIntrinsicOperationV1::FabsF32 => {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "compiler intrinsic has no fill-profile lowering rule",
                ));
            }
            SemanticCompilerIntrinsicOperationV1::Trap => {
                unreachable!("trap compiler intrinsic returned before destination lowering")
            }
        };
        self.store_enum_payload_v1(
            block,
            None,
            destination.place().local(),
            &binding,
            operations,
        )?;
        self.bind_destination(block, None, destination.place(), binding)?;
        Ok(Terminator::Branch {
            target: BlockId(destination.edge().target().index()),
            arguments: self.edge_arguments(block, destination.edge().target(), operations)?,
        })
    }

    fn lower_subgroup_reduce_f32(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
        value: SemanticValueBindingV1,
        width: u32,
        kind: SemanticSubgroupReductionKindV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        if width == 0 || !width.is_power_of_two() || width > 64 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "subgroup reduction width must be a power of two in 1..=64",
            ));
        }
        let (mut reduced, ty) = value
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if ty != Type::Scalar(ScalarType::F32) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "subgroup reduction input is not f32",
            ));
        }
        let (lane, _) = self
            .emit(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Wave(WaveOperation::full(
                    WaveOperationKind::LaneId,
                    WaveWidth::Wave64,
                )),
            )?
            .value()
            .expect("emitted lane id");
        let width_value = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(width)),
        )?;
        let local_lane = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Binary {
                op: BinaryOp::Remainder,
                lhs: lane,
                rhs: width_value,
            },
        )?;
        let subgroup = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Binary {
                op: BinaryOp::Divide,
                lhs: lane,
                rhs: width_value,
            },
        )?;
        let subgroup_base = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Binary {
                op: BinaryOp::Multiply,
                lhs: subgroup,
                rhs: width_value,
            },
        )?;

        let mut offset = width / 2;
        while offset != 0 {
            let offset_value = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(offset)),
            )?;
            let source_local = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs: local_lane,
                    rhs: offset_value,
                },
            )?;
            let source_lane = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: subgroup_base,
                    rhs: source_local,
                },
            )?;
            let bits = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: reduced,
                    to: Type::Scalar(ScalarType::U32),
                },
            )?;
            let peer_bits = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Wave(WaveOperation::full(
                    WaveOperationKind::ShuffleIndex {
                        value: bits,
                        source_lane,
                        tile_width: width,
                    },
                    WaveWidth::Wave64,
                )),
            )?;
            let peer = self.emit_id(
                operations,
                Type::Scalar(ScalarType::F32),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: peer_bits,
                    to: Type::Scalar(ScalarType::F32),
                },
            )?;
            reduced = match kind {
                SemanticSubgroupReductionKindV1::Sum => self.emit_id(
                    operations,
                    Type::Scalar(ScalarType::F32),
                    OperationKind::Binary {
                        op: BinaryOp::Add,
                        lhs: reduced,
                        rhs: peer,
                    },
                )?,
                SemanticSubgroupReductionKindV1::Maximum => {
                    let take_peer = self.emit_id(
                        operations,
                        Type::BOOL,
                        OperationKind::Compare {
                            predicate: ComparePredicate::LessThan,
                            lhs: reduced,
                            rhs: peer,
                        },
                    )?;
                    self.emit_id(
                        operations,
                        Type::Scalar(ScalarType::F32),
                        OperationKind::Select {
                            condition: take_peer,
                            true_value: peer,
                            false_value: reduced,
                        },
                    )?
                }
            };
            offset /= 2;
        }
        Ok(SemanticValueBindingV1::Value {
            id: reduced,
            ty: Type::Scalar(ScalarType::F32),
        })
    }

    fn lower_checked_strided_read_view(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        result_type: SemanticTypeIdV1,
        view_type: SemanticTypeIdV1,
        error_type: SemanticTypeIdV1,
        element_type: Type,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 5)?;
        let bits = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let (bits, bits_ty) = bits
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let Type::Slice(slice) = &bits_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked strided view storage is not a slice",
            ));
        };
        if slice.address_space != AddressSpace::Global
            || slice.access != AccessMode::ReadOnly
            || *slice.element != element_type
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked strided view storage must be a read-only global scalar slice",
            ));
        }
        let mut indices = Vec::with_capacity(4);
        for argument in &call.arguments()[1..] {
            let binding = self.lower_operand(block, None, argument, operations)?;
            indices.push(
                self.coerce_index(block, operations, binding)?
                    .value()
                    .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
                    .0,
            );
        }
        let [offset, rows, columns, stride] = indices
            .try_into()
            .expect("four checked row-major view indices");
        let zero = self.emit_index_constant(operations, 0)?;
        let one = self.emit_index_constant(operations, 1)?;
        let rows_zero = self.emit_compare(operations, ComparePredicate::Equal, rows, zero)?;
        let columns_zero = self.emit_compare(operations, ComparePredicate::Equal, columns, zero)?;
        let empty = self.emit_bool_or(operations, rows_zero, columns_zero)?;
        let stride_wide_enough = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            columns,
            stride,
        )?;
        let stride_valid = self.emit_bool_or(operations, empty, stride_wide_enough)?;

        let (rows_minus_one, rows_safe) =
            self.emit_checked_index(operations, CheckedBinaryOperator::Subtract, rows, one)?;
        let (row_extent, multiply_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Multiply,
            rows_minus_one,
            stride,
        )?;
        let (matrix_extent, columns_safe) =
            self.emit_checked_index(operations, CheckedBinaryOperator::Add, row_extent, columns)?;
        let (required_nonempty, offset_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Add,
            offset,
            matrix_extent,
        )?;
        let arithmetic_safe = self.emit_bool_and(operations, rows_safe, multiply_safe)?;
        let arithmetic_safe = self.emit_bool_and(operations, arithmetic_safe, columns_safe)?;
        let arithmetic_safe = self.emit_bool_and(operations, arithmetic_safe, offset_safe)?;
        let arithmetic_safe = self.emit_bool_or(operations, empty, arithmetic_safe)?;
        let required = self.emit_select_index(operations, empty, offset, required_nonempty)?;
        let length = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::SliceLength { slice: bits },
        )?;
        let in_bounds = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            required,
            length,
        )?;
        let valid = self.emit_bool_and(operations, stride_valid, arithmetic_safe)?;
        let valid = self.emit_bool_and(operations, valid, in_bounds)?;

        let (discriminant_type, variants) = semantic_enum_shape(self.types, result_type)?;
        let ok_variant = unique_enum_variant_with_field(variants, view_type).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "checked strided view Result has no unique Ok payload",
            )
        })?;
        let error_variant =
            unique_enum_variant_with_field(variants, error_type).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "checked strided view Result has no unique error payload",
                )
            })?;
        let (error_discriminant_type, error_variants) =
            semantic_enum_shape(self.types, error_type)?;
        // Provider-bound checked views end in InvalidStride, ExtentOverflow,
        // and OutOfBounds; preceding unit variants describe statically
        // impossible element-layout failures.
        let out_of_bounds_field_types = error_variants
            .last()
            .map(|variant| {
                variant
                    .fields()
                    .fields()
                    .iter()
                    .map(|field| lower_scalar_type(self.types, *field))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        if error_variants.len() < 3
            || error_variants[..error_variants.len() - 3]
                .iter()
                .any(|variant| !variant.fields().fields().is_empty())
            || !error_variants[error_variants.len() - 3]
                .fields()
                .fields()
                .is_empty()
            || !error_variants[error_variants.len() - 2]
                .fields()
                .fields()
                .is_empty()
            || error_variants[error_variants.len() - 1]
                .fields()
                .fields()
                .len()
                != 2
            || !matches!(out_of_bounds_field_types.as_deref(), Some([first, second])
                if (first == &Type::INDEX
                    || index_and_u64_are_transport_equivalent(&Type::INDEX, first))
                    && (second == &Type::INDEX
                        || index_and_u64_are_transport_equivalent(&Type::INDEX, second)))
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked strided view error type does not match the provider-bound unit/unit/out-of-bounds tail",
            ));
        }
        let discriminant_ty = lower_scalar_type(self.types, discriminant_type)?;
        let ok_discriminant = self.emit_id(
            operations,
            discriminant_ty.clone(),
            OperationKind::Constant(integer_constant(
                &discriminant_ty,
                variants[ok_variant as usize].discriminant(),
            )?),
        )?;
        let error_discriminant = self.emit_id(
            operations,
            discriminant_ty.clone(),
            OperationKind::Constant(integer_constant(
                &discriminant_ty,
                variants[error_variant as usize].discriminant(),
            )?),
        )?;
        let discriminant = self.emit_id(
            operations,
            discriminant_ty.clone(),
            OperationKind::Select {
                condition: valid,
                true_value: ok_discriminant,
                false_value: error_discriminant,
            },
        )?;
        let error_discriminant_ty = lower_scalar_type(self.types, error_discriminant_type)?;
        let invalid_stride_variant = error_variants.len() - 3;
        let extent_overflow_variant = error_variants.len() - 2;
        let out_of_bounds_variant = error_variants.len() - 1;
        let error_discriminant = |this: &mut Self,
                                  operations: &mut Vec<Operation>,
                                  variant: usize|
         -> Result<ValueId, ProductionSemanticKirErrorV1> {
            this.emit_id(
                operations,
                error_discriminant_ty.clone(),
                OperationKind::Constant(integer_constant(
                    &error_discriminant_ty,
                    error_variants[variant].discriminant(),
                )?),
            )
        };
        let invalid_stride_discriminant =
            error_discriminant(self, operations, invalid_stride_variant)?;
        let extent_overflow_discriminant =
            error_discriminant(self, operations, extent_overflow_variant)?;
        let out_of_bounds_discriminant =
            error_discriminant(self, operations, out_of_bounds_variant)?;
        let arithmetic_error_discriminant = self.emit_id(
            operations,
            error_discriminant_ty.clone(),
            OperationKind::Select {
                condition: arithmetic_safe,
                true_value: out_of_bounds_discriminant,
                false_value: extent_overflow_discriminant,
            },
        )?;
        let selected_error_discriminant = self.emit_id(
            operations,
            error_discriminant_ty.clone(),
            OperationKind::Select {
                condition: stride_valid,
                true_value: arithmetic_error_discriminant,
                false_value: invalid_stride_discriminant,
            },
        )?;
        let [required_field_ty, actual_field_ty] = out_of_bounds_field_types
            .as_deref()
            .expect("validated checked-view out-of-bounds payload")
        else {
            unreachable!("validated checked-view out-of-bounds payload has two fields")
        };
        let retain_index = |this: &mut Self,
                            operations: &mut Vec<Operation>,
                            value: ValueId,
                            target: &Type|
         -> Result<ValueId, ProductionSemanticKirErrorV1> {
            if target == &Type::INDEX {
                Ok(value)
            } else {
                this.emit_id(
                    operations,
                    target.clone(),
                    OperationKind::Cast {
                        kind: CastKind::Bitcast,
                        value,
                        to: target.clone(),
                    },
                )
            }
        };
        let required_payload = retain_index(self, operations, required, required_field_ty)?;
        let actual_payload = retain_index(self, operations, length, actual_field_ty)?;
        let view = SemanticValueBindingV1::Aggregate(vec![
            SemanticValueBindingV1::Value {
                id: bits,
                ty: bits_ty,
            },
            SemanticValueBindingV1::Value {
                id: offset,
                ty: Type::INDEX,
            },
            SemanticValueBindingV1::Value {
                id: rows,
                ty: Type::INDEX,
            },
            SemanticValueBindingV1::Value {
                id: columns,
                ty: Type::INDEX,
            },
            SemanticValueBindingV1::Value {
                id: stride,
                ty: Type::INDEX,
            },
        ]);
        let mut error_payloads = BTreeMap::new();
        for variant in 0..error_variants.len() - 1 {
            error_payloads.insert(variant as u32, Vec::new());
        }
        error_payloads.insert(
            out_of_bounds_variant as u32,
            vec![
                SemanticValueBindingV1::Value {
                    id: required_payload,
                    ty: required_field_ty.clone(),
                },
                SemanticValueBindingV1::Value {
                    id: actual_payload,
                    ty: actual_field_ty.clone(),
                },
            ],
        );
        let error = SemanticValueBindingV1::Enum {
            discriminant: selected_error_discriminant,
            discriminant_ty: error_discriminant_ty,
            semantic_type: error_type,
            variant: None,
            payloads: error_payloads,
        };
        Ok(SemanticValueBindingV1::Enum {
            discriminant,
            discriminant_ty,
            semantic_type: result_type,
            variant: None,
            payloads: BTreeMap::from([(ok_variant, vec![view]), (error_variant, vec![error])]),
        })
    }

    fn lower_strided_read_view_load_or(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        element_type: Type,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 4)?;
        let view = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let view = view
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if view.len() != 5 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "strided read load view has no exact checked representation",
            ));
        }
        let (data, data_ty) = view[0].clone();
        let Type::Slice(slice) = &data_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "strided read load storage is not a slice",
            ));
        };
        if slice.address_space != AddressSpace::Global
            || slice.access != AccessMode::ReadOnly
            || *slice.element != element_type
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "strided read load storage contract changed",
            ));
        }
        let [(_, _), (offset, _), (rows, _), (columns, _), (stride, _)] = view.as_slice() else {
            unreachable!("checked five-component strided view");
        };
        let row = self.lower_operand(block, None, &call.arguments()[1], operations)?;
        let row = self
            .coerce_index(block, operations, row)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let column = self.lower_operand(block, None, &call.arguments()[2], operations)?;
        let column = self
            .coerce_index(block, operations, column)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let fallback = self.lower_operand(block, None, &call.arguments()[3], operations)?;
        let (fallback, fallback_ty) = fallback
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if fallback_ty != element_type {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "strided read fallback element type changed",
            ));
        }

        let row_valid = self.emit_compare(operations, ComparePredicate::LessThan, row, *rows)?;
        let column_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, column, *columns)?;
        let (row_offset, row_safe) =
            self.emit_checked_index(operations, CheckedBinaryOperator::Multiply, row, *stride)?;
        let (index, offset_safe) =
            self.emit_checked_index(operations, CheckedBinaryOperator::Add, *offset, row_offset)?;
        let (index, column_safe) =
            self.emit_checked_index(operations, CheckedBinaryOperator::Add, index, column)?;
        let length = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::SliceLength { slice: data },
        )?;
        let index_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, index, length)?;
        let mut predicate = self.emit_bool_and(operations, row_valid, column_valid)?;
        predicate = self.emit_bool_and(operations, predicate, row_safe)?;
        predicate = self.emit_bool_and(operations, predicate, offset_safe)?;
        predicate = self.emit_bool_and(operations, predicate, column_safe)?;
        predicate = self.emit_bool_and(operations, predicate, index_valid)?;
        let zero = self.emit_index_constant(operations, 0)?;
        let safe_index = self.emit_select_index(operations, predicate, index, zero)?;
        let base = self.emit_id(
            operations,
            Type::pointer(element_type.clone(), slice.address_space, slice.access),
            OperationKind::SliceData { slice: data },
        )?;
        let pointer = self.emit_id(
            operations,
            Type::pointer(element_type.clone(), slice.address_space, slice.access),
            OperationKind::GetElementPointer {
                base,
                offset: safe_index,
            },
        )?;
        let alignment = strided_read_scalar_alignment_v1(&element_type).ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "strided read element has no supported scalar alignment",
            )
        })?;
        let value = self.emit_id(
            operations,
            element_type.clone(),
            OperationKind::GuardedLoad {
                pointer,
                predicate,
                fallback,
                access: MemoryAccess::new(slice.address_space, alignment),
            },
        )?;
        Ok(SemanticValueBindingV1::Value {
            id: value,
            ty: element_type,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_bf16_matrix_load(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 4)?;
        let view = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let view = view
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if view.len() != 5 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "BF16 matrix load view has no exact checked representation",
            ));
        }
        let (bits, bits_ty) = view[0].clone();
        let Type::Slice(slice) = &bits_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "BF16 matrix load view storage is not a slice",
            ));
        };
        if *slice.element != Type::Scalar(ScalarType::U16) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "BF16 matrix load view element type changed",
            ));
        }
        let [(_, _), (offset, _), (rows, _), (columns, _), (stride, _)] = view.as_slice() else {
            unreachable!("checked five-component view");
        };
        let offset = *offset;
        let rows = *rows;
        let columns = *columns;
        let stride = *stride;
        let (lane, wave) = require_current_wave_lane(
            block,
            self.lower_operand(block, None, &call.arguments()[1], operations)?,
            contract.wave_width,
            "typed matrix load lane",
        )?;
        let lane_index = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Cast {
                kind: CastKind::ZeroExtend,
                value: lane,
                to: Type::INDEX,
            },
        )?;
        let first_base = self.lower_operand(block, None, &call.arguments()[2], operations)?;
        let first_base = self
            .coerce_index(block, operations, first_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let second_base = self.lower_operand(block, None, &call.arguments()[3], operations)?;
        let second_base = self
            .coerce_index(block, operations, second_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let fifteen = self.emit_index_constant(operations, 15)?;
        let sixteen = self.emit_index_constant(operations, 16)?;
        let four = self.emit_index_constant(operations, 4)?;
        let lane_minor =
            self.emit_index_binary(operations, BinaryOp::BitAnd, lane_index, fifteen)?;
        let lane_group =
            self.emit_index_binary(operations, BinaryOp::Divide, lane_index, sixteen)?;
        let lane_group =
            self.emit_index_binary(operations, BinaryOp::Multiply, lane_group, four)?;
        let bases = semantic_mfma_operand_bases_v1(contract.role, first_base, second_base);
        let (row_or_column, minor_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Add,
            bases.minor,
            lane_minor,
        )?;
        let (first_reduction, reduction_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Add,
            bases.reduction,
            lane_group,
        )?;
        let mut present = self.emit_bool_and(operations, minor_safe, reduction_safe)?;
        let data = self.emit_id(
            operations,
            Type::pointer(
                Type::Scalar(ScalarType::U16),
                slice.address_space,
                slice.access,
            ),
            OperationKind::SliceData { slice: bits },
        )?;
        let length = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::SliceLength { slice: bits },
        )?;
        let zero_index = self.emit_index_constant(operations, 0)?;
        let zero_bits = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U16),
            OperationKind::Constant(Constant::U16(0)),
        )?;
        let component_count = contract.profile.operand_components_per_lane();
        let mut values = Vec::with_capacity(component_count);
        for component in 0..u64::try_from(component_count).expect("MFMA component count fits u64") {
            let component_value = self.emit_index_constant(operations, component)?;
            let (reduction, component_safe) = self.emit_checked_index(
                operations,
                CheckedBinaryOperator::Add,
                first_reduction,
                component_value,
            )?;
            present = self.emit_bool_and(operations, present, component_safe)?;
            let (row, column) = match contract.role {
                fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1::A => {
                    (row_or_column, reduction)
                }
                fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1::B => {
                    (reduction, row_or_column)
                }
            };
            let row_valid = self.emit_compare(operations, ComparePredicate::LessThan, row, rows)?;
            let column_valid =
                self.emit_compare(operations, ComparePredicate::LessThan, column, columns)?;
            let (row_offset, row_safe) =
                self.emit_checked_index(operations, CheckedBinaryOperator::Multiply, row, stride)?;
            let (index, offset_safe) = self.emit_checked_index(
                operations,
                CheckedBinaryOperator::Add,
                offset,
                row_offset,
            )?;
            let (index, column_safe) =
                self.emit_checked_index(operations, CheckedBinaryOperator::Add, index, column)?;
            let index_in_bounds =
                self.emit_compare(operations, ComparePredicate::LessThan, index, length)?;
            let mut guard = self.emit_bool_and(operations, present, row_valid)?;
            guard = self.emit_bool_and(operations, guard, column_valid)?;
            guard = self.emit_bool_and(operations, guard, row_safe)?;
            guard = self.emit_bool_and(operations, guard, offset_safe)?;
            guard = self.emit_bool_and(operations, guard, column_safe)?;
            guard = self.emit_bool_and(operations, guard, index_in_bounds)?;
            let safe_index = self.emit_select_index(operations, guard, index, zero_index)?;
            let pointer = self.emit_id(
                operations,
                Type::pointer(
                    Type::Scalar(ScalarType::U16),
                    slice.address_space,
                    slice.access,
                ),
                OperationKind::GetElementPointer {
                    base: data,
                    offset: safe_index,
                },
            )?;
            let loaded = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U16),
                OperationKind::GuardedLoad {
                    pointer,
                    predicate: guard,
                    fallback: zero_bits,
                    access: MemoryAccess::new(slice.address_space, 2),
                },
            )?;
            let value = self.emit_id(
                operations,
                Type::Scalar(ScalarType::Bf16),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: loaded,
                    to: Type::Scalar(ScalarType::Bf16),
                },
            )?;
            values.push((value, Type::Scalar(ScalarType::Bf16)));
        }
        Ok(SemanticValueBindingV1::MatrixFragment {
            values,
            contract,
            storage_layout,
            wave,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_gfx950_low_precision_matrix_load(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let fp4 = contract.profile == SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128;
        if !fp4 && contract.profile != SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
            || contract.register_distribution
                != SemanticMfmaRegisterDistributionV1::Gfx950M16N16K128
            || storage_layout != SemanticMfmaStorageLayoutV1::RowMajor
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 low-precision load profile or distribution changed",
            ));
        }
        self.require_call_argument_count(block, call, 4)?;
        let view = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let view = view
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let [
            (_, bits_ty),
            (offset, _),
            (rows, _),
            (columns, _),
            (stride, _),
        ] = view.as_slice()
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 low-precision matrix load view has no exact checked representation",
            ));
        };
        let (bits, _) = view[0].clone();
        let Type::Slice(slice) = bits_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 low-precision matrix view storage is not a slice",
            ));
        };
        if *slice.element != Type::Scalar(ScalarType::U8) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 low-precision matrix view element type changed",
            ));
        }
        let offset = *offset;
        let rows = *rows;
        let columns = *columns;
        let stride = *stride;
        let (lane, wave) = require_current_wave_lane(
            block,
            self.lower_operand(block, None, &call.arguments()[1], operations)?,
            contract.wave_width,
            "gfx950 low-precision matrix load lane",
        )?;
        let lane_index = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Cast {
                kind: CastKind::ZeroExtend,
                value: lane,
                to: Type::INDEX,
            },
        )?;
        let first_base = self.lower_operand(block, None, &call.arguments()[2], operations)?;
        let first_base = self
            .coerce_index(block, operations, first_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let second_base = self.lower_operand(block, None, &call.arguments()[3], operations)?;
        let second_base = self
            .coerce_index(block, operations, second_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;

        let fifteen = self.emit_index_constant(operations, 15)?;
        let sixteen = self.emit_index_constant(operations, 16)?;
        let lane_minor =
            self.emit_index_binary(operations, BinaryOp::BitAnd, lane_index, fifteen)?;
        let lane_group =
            self.emit_index_binary(operations, BinaryOp::Divide, lane_index, sixteen)?;
        let group_width = self.emit_index_constant(operations, if fp4 { 32 } else { 16 })?;
        let lane_group =
            self.emit_index_binary(operations, BinaryOp::Multiply, lane_group, group_width)?;
        let bases = semantic_mfma_operand_bases_v1(contract.role, first_base, second_base);
        let (row_or_column, minor_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Add,
            bases.minor,
            lane_minor,
        )?;
        let (group_reduction, reduction_safe) = self.emit_checked_index(
            operations,
            CheckedBinaryOperator::Add,
            bases.reduction,
            lane_group,
        )?;
        let present = self.emit_bool_and(operations, minor_safe, reduction_safe)?;
        let data = self.emit_id(
            operations,
            Type::pointer(
                Type::Scalar(ScalarType::U8),
                slice.address_space,
                slice.access,
            ),
            OperationKind::SliceData { slice: bits },
        )?;
        let length = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::SliceLength { slice: bits },
        )?;
        let zero_index = self.emit_index_constant(operations, 0)?;
        let zero_u8 = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U8),
            OperationKind::Constant(Constant::U8(0)),
        )?;
        let zero_u32 = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(0)),
        )?;
        let fifteen_u32 = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(15)),
        )?;
        let mut values = Vec::with_capacity(8);
        let mut packed = zero_u32;
        for component in 0_u64..32 {
            let split_k = if fp4 {
                component
            } else {
                (component % 16) + (component / 16) * 64
            };
            let split_k = self.emit_index_constant(operations, split_k)?;
            let (reduction, component_safe) = self.emit_checked_index(
                operations,
                CheckedBinaryOperator::Add,
                group_reduction,
                split_k,
            )?;
            let component_present = self.emit_bool_and(operations, present, component_safe)?;
            let (row, column) = match contract.role {
                SemanticMfmaOperandRoleV1::A => (row_or_column, reduction),
                SemanticMfmaOperandRoleV1::B => (reduction, row_or_column),
            };
            let row_valid = self.emit_compare(operations, ComparePredicate::LessThan, row, rows)?;
            let column_valid =
                self.emit_compare(operations, ComparePredicate::LessThan, column, columns)?;
            let (row_offset, row_safe) =
                self.emit_checked_index(operations, CheckedBinaryOperator::Multiply, row, stride)?;
            let (index, offset_safe) = self.emit_checked_index(
                operations,
                CheckedBinaryOperator::Add,
                offset,
                row_offset,
            )?;
            let (index, column_safe) =
                self.emit_checked_index(operations, CheckedBinaryOperator::Add, index, column)?;
            let index_in_bounds =
                self.emit_compare(operations, ComparePredicate::LessThan, index, length)?;
            let mut guard = self.emit_bool_and(operations, component_present, row_valid)?;
            guard = self.emit_bool_and(operations, guard, column_valid)?;
            guard = self.emit_bool_and(operations, guard, row_safe)?;
            guard = self.emit_bool_and(operations, guard, offset_safe)?;
            guard = self.emit_bool_and(operations, guard, column_safe)?;
            guard = self.emit_bool_and(operations, guard, index_in_bounds)?;
            let safe_index = self.emit_select_index(operations, guard, index, zero_index)?;
            let pointer = self.emit_id(
                operations,
                Type::pointer(
                    Type::Scalar(ScalarType::U8),
                    slice.address_space,
                    slice.access,
                ),
                OperationKind::GetElementPointer {
                    base: data,
                    offset: safe_index,
                },
            )?;
            let loaded = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U8),
                OperationKind::GuardedLoad {
                    pointer,
                    predicate: guard,
                    fallback: zero_u8,
                    access: MemoryAccess::new(slice.address_space, 1),
                },
            )?;
            let mut widened = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Cast {
                    kind: CastKind::ZeroExtend,
                    value: loaded,
                    to: Type::Scalar(ScalarType::U32),
                },
            )?;
            if fp4 {
                widened = self.emit_id(
                    operations,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Binary {
                        op: BinaryOp::BitAnd,
                        lhs: widened,
                        rhs: fifteen_u32,
                    },
                )?;
            }
            let shift = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(if fp4 {
                    ((component % 8) * 4) as u32
                } else {
                    ((component % 4) * 8) as u32
                })),
            )?;
            let shifted = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Binary {
                    op: BinaryOp::ShiftLeft,
                    lhs: widened,
                    rhs: shift,
                },
            )?;
            packed = self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Binary {
                    op: BinaryOp::BitOr,
                    lhs: packed,
                    rhs: shifted,
                },
            )?;
            if component % (if fp4 { 8 } else { 4 }) == (if fp4 { 7 } else { 3 }) {
                values.push((packed, Type::Scalar(ScalarType::U32)));
                packed = zero_u32;
            }
        }
        while values.len() < 8 {
            values.push((zero_u32, Type::Scalar(ScalarType::U32)));
        }
        Ok(SemanticValueBindingV1::MatrixFragment {
            values,
            contract,
            storage_layout,
            wave,
        })
    }

    fn lower_gfx950_lds_transpose_tile_operand(
        &mut self,
        block: SemanticBlockIdV1,
        operand: &SemanticOperandV1,
        operations: &mut Vec<Operation>,
        expected_type: SemanticTypeIdV1,
        expected_format: SemanticGfx950LdsTransposeFormatV1,
        expected_state: SemanticGfx950LdsTransposeStateV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        if semantic_operand_type(operand) != expected_type {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose receiver type differs from its intrinsic contract",
            ));
        }
        if !matches!(
            operand,
            SemanticOperandV1::Constant(constant)
                if matches!(constant.value(), SemanticConstantValueV1::ZeroSized)
        ) {
            return self.lower_operand(block, None, operand, operations);
        }

        let mut candidates = self.locals.iter().filter_map(|binding| match binding {
            Some(
                binding @ SemanticValueBindingV1::Gfx950LdsTransposeTile { format, state, .. },
            ) if *format == expected_format && *state == expected_state => Some(binding.clone()),
            _ => None,
        });
        let candidate = candidates.next().ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose ZST receiver has no live authenticated state token",
            )
        })?;
        if candidates.next().is_some() {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose ZST receiver has ambiguous authenticated state tokens",
            ));
        }
        Ok(candidate)
    }

    fn lower_gfx950_lds_transpose_current(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        format: SemanticGfx950LdsTransposeFormatV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let _ = require_current_wave_lane(
            block,
            self.lower_operand(block, None, &call.arguments()[0], operations)?,
            64,
            "gfx950 LDS transpose current lane",
        )?;
        let format = lower_gfx950_lds_transpose_format_v1(format);
        let results = self.emit_results(
            operations,
            vec![gfx950_lds_transpose_pointer_type_v1()],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Current { format },
            )),
        )?;
        Ok(SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage: results[0].id,
            format: semantic_gfx950_lds_transpose_format_v1(format),
            state: SemanticGfx950LdsTransposeStateV1::Uninitialized,
        })
    }

    fn lower_gfx950_lds_transpose_stage(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_tile_type: SemanticTypeIdV1,
        format: SemanticGfx950LdsTransposeFormatV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 4)?;
        let tile = self.lower_gfx950_lds_transpose_tile_operand(
            block,
            &call.arguments()[0],
            operations,
            input_tile_type,
            format,
            SemanticGfx950LdsTransposeStateV1::Uninitialized,
        )?;
        let SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage,
            format: actual_format,
            state: SemanticGfx950LdsTransposeStateV1::Uninitialized,
        } = tile
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose stage requires an uninitialized tile",
            ));
        };
        if actual_format != format {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose stage format differs from its tile",
            ));
        }
        let view = self
            .lower_operand(block, None, &call.arguments()[1], operations)?
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let [
            (source_slice, source_ty),
            (offset, offset_ty),
            (rows, rows_ty),
            (columns, columns_ty),
            (stride, stride_ty),
        ] = view.as_slice()
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose stage view has no exact checked representation",
            ));
        };
        let Type::Slice(slice) = source_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose stage source is not a slice",
            ));
        };
        if *slice.element != Type::Scalar(ScalarType::U8)
            || slice.address_space != AddressSpace::Global
            || slice.access != AccessMode::ReadOnly
            || [offset_ty, rows_ty, columns_ty, stride_ty]
                .iter()
                .any(|ty| **ty != Type::INDEX)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose stage source view type changed",
            ));
        }
        let token_base = self.lower_operand(block, None, &call.arguments()[2], operations)?;
        let token_base = self
            .coerce_index(block, operations, token_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let reduction_base = self.lower_operand(block, None, &call.arguments()[3], operations)?;
        let reduction_base = self
            .coerce_index(block, operations, reduction_base)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?
            .0;
        let format = lower_gfx950_lds_transpose_format_v1(format);
        let results = self.emit_results(
            operations,
            vec![gfx950_lds_transpose_pointer_type_v1()],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Stage {
                    format,
                    storage,
                    source_slice: *source_slice,
                    offset: *offset,
                    rows: *rows,
                    columns: *columns,
                    stride: *stride,
                    token_base,
                    reduction_base,
                },
            )),
        )?;
        Ok(SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage: results[0].id,
            format: semantic_gfx950_lds_transpose_format_v1(format),
            state: SemanticGfx950LdsTransposeStateV1::Staged,
        })
    }

    fn lower_gfx950_lds_transpose_publish(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_tile_type: SemanticTypeIdV1,
        format: SemanticGfx950LdsTransposeFormatV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let tile = self.lower_gfx950_lds_transpose_tile_operand(
            block,
            &call.arguments()[0],
            operations,
            input_tile_type,
            format,
            SemanticGfx950LdsTransposeStateV1::Staged,
        )?;
        let SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage,
            format: actual_format,
            state: SemanticGfx950LdsTransposeStateV1::Staged,
        } = tile
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose publish requires a staged tile",
            ));
        };
        if actual_format != format {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose publish format differs from its tile",
            ));
        }
        let format = lower_gfx950_lds_transpose_format_v1(format);
        let results = self.emit_results(
            operations,
            vec![gfx950_lds_transpose_pointer_type_v1()],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Publish { format, storage },
            )),
        )?;
        Ok(SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage: results[0].id,
            format: semantic_gfx950_lds_transpose_format_v1(format),
            state: SemanticGfx950LdsTransposeStateV1::Published,
        })
    }

    fn lower_gfx950_lds_transpose_read(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        tile_type: SemanticTypeIdV1,
        contract: SemanticMfmaOperandContractV1,
        format: SemanticGfx950LdsTransposeFormatV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let tile = self.lower_gfx950_lds_transpose_tile_operand(
            block,
            &call.arguments()[0],
            operations,
            tile_type,
            format,
            SemanticGfx950LdsTransposeStateV1::Published,
        )?;
        let SemanticValueBindingV1::Gfx950LdsTransposeTile {
            storage,
            format: actual_format,
            state: SemanticGfx950LdsTransposeStateV1::Published,
        } = tile
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose read requires a published tile",
            ));
        };
        let expected_profile = match format {
            SemanticGfx950LdsTransposeFormatV1::Fp4E2M1 => {
                SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
            }
            SemanticGfx950LdsTransposeFormatV1::Fp8E4M3 => {
                SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
            }
        };
        if actual_format != format
            || contract.role != SemanticMfmaOperandRoleV1::B
            || contract.profile != expected_profile
            || contract.register_distribution
                != SemanticMfmaRegisterDistributionV1::Gfx950M16N16K128
            || contract.wave_width != 64
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "gfx950 LDS transpose read contract differs from its published tile",
            ));
        }
        let format = lower_gfx950_lds_transpose_format_v1(format);
        let results = self.emit_results(
            operations,
            vec![Type::Scalar(ScalarType::U32); 8],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Read { format, storage },
            )),
        )?;
        Ok(SemanticValueBindingV1::MatrixFragment {
            values: results
                .into_iter()
                .map(|result| (result.id, result.ty))
                .collect(),
            contract,
            storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
            wave: SemanticCurrentWaveV1::new(64),
        })
    }

    fn require_call_argument_count(
        &self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        expected: usize,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if call.arguments().len() == expected {
            Ok(())
        } else {
            Err(unsupported(
                0,
                Some(block.index()),
                None,
                "compiler intrinsic argument count changed",
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_checked_shift(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_space: SemanticDisjointIndexSpaceV1,
        output_space: SemanticDisjointIndexSpaceV1,
        offset: u64,
        input_is_disjoint: bool,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let binding = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let SemanticValueBindingV1::IndexWitness {
            id,
            index_space: actual,
            disjoint,
            ..
        } = binding
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_shift receiver is not index authority",
            ));
        };
        if actual != input_space || disjoint != input_is_disjoint {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_shift input mapping identity changed",
            ));
        }
        let expected_output = match input_space {
            SemanticDisjointIndexSpaceV1::Index1d => {
                SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset }
            }
            SemanticDisjointIndexSpaceV1::ShiftedIndex1d { .. }
            | SemanticDisjointIndexSpaceV1::BlockedIndex1d { .. }
            | SemanticDisjointIndexSpaceV1::Tiled2dIndex1d { .. }
            | SemanticDisjointIndexSpaceV1::RowStriped2dIndex1d { .. }
            | SemanticDisjointIndexSpaceV1::GridExclusive => {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "checked_shift input mapping is unsupported",
                ));
            }
        };
        if output_space != expected_output {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_shift output mapping identity changed",
            ));
        }
        let (maximum, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(u64::MAX - offset)),
            )?
            .value()
            .expect("emitted index constant");
        let (present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThanOrEqual,
                    lhs: id,
                    rhs: maximum,
                },
            )?
            .value()
            .expect("emitted checked-shift predicate");
        let (offset_value, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(offset)),
            )?
            .value()
            .expect("emitted index constant");
        let (shifted, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: id,
                    rhs: offset_value,
                },
            )?
            .value()
            .expect("emitted shifted index");
        Ok(SemanticValueBindingV1::OptionIndexWitness {
            present,
            availability: self
                .option_dominance
                .availability(
                    call.destination()
                        .expect("checked destination")
                        .place()
                        .local(),
                )
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "checked-shift Option lacks an authenticated Some edge",
                    )
                })?,
            id: shifted,
            index_space: output_space,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_checked_block(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_space: SemanticDisjointIndexSpaceV1,
        output_space: SemanticDisjointIndexSpaceV1,
        lanes_per_block: u64,
        elements_per_lane: u64,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let input = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let SemanticValueBindingV1::IndexWitness {
            id: raw,
            index_space: actual,
            disjoint: false,
            availability: None,
        } = input
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_block receiver is not thread-index authority",
            ));
        };
        let expected = SemanticDisjointIndexSpaceV1::BlockedIndex1d {
            lanes_per_block,
            elements_per_lane,
        };
        let Some(block_elements) = lanes_per_block.checked_mul(elements_per_lane) else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_block dimensions overflow",
            ));
        };
        if actual != input_space
            || input_space != SemanticDisjointIndexSpaceV1::Index1d
            || output_space != expected
            || lanes_per_block == 0
            || elements_per_lane == 0
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_block mapping identity is malformed",
            ));
        }
        let (lanes, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(lanes_per_block)),
            )?
            .value()
            .expect("emitted lanes constant");
        let (block_elements_value, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(block_elements)),
            )?
            .value()
            .expect("emitted block-elements constant");
        let (block_index, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    lhs: raw,
                    rhs: lanes,
                },
            )?
            .value()
            .expect("emitted block quotient");
        let (lane, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs: raw,
                    rhs: lanes,
                },
            )?
            .value()
            .expect("emitted lane remainder");
        let (maximum_block, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(u64::MAX / block_elements)),
            )?
            .value()
            .expect("emitted maximum block");
        let (block_safe, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThanOrEqual,
                    lhs: block_index,
                    rhs: maximum_block,
                },
            )?
            .value()
            .expect("emitted block overflow predicate");
        let (block_base, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Multiply,
                    lhs: block_index,
                    rhs: block_elements_value,
                },
            )?
            .value()
            .expect("emitted block base");
        let final_component_base = (elements_per_lane - 1) * lanes_per_block;
        let (final_component, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(final_component_base)),
            )?
            .value()
            .expect("emitted final-component base");
        let (final_offset, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: final_component,
                    rhs: lane,
                },
            )?
            .value()
            .expect("emitted final-component offset");
        let (final_index, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: block_base,
                    rhs: final_offset,
                },
            )?
            .value()
            .expect("emitted final blocked index");
        let (sum_safe, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThanOrEqual,
                    lhs: block_base,
                    rhs: final_index,
                },
            )?
            .value()
            .expect("emitted blocked sum predicate");
        let (present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: block_safe,
                    rhs: sum_safe,
                },
            )?
            .value()
            .expect("emitted checked-block predicate");
        Ok(SemanticValueBindingV1::OptionComponentWitness {
            present,
            raw,
            index_space: expected,
            availability: self
                .option_dominance
                .availability(
                    call.destination()
                        .expect("checked destination")
                        .place()
                        .local(),
                )
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "checked-block Option lacks an authenticated Some edge",
                    )
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_checked_tiled_2d(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_space: SemanticDisjointIndexSpaceV1,
        output_space: SemanticDisjointIndexSpaceV1,
        lanes_per_tile: u64,
        tile_rows: u64,
        tile_columns: u64,
        elements_per_lane: u64,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let input = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let SemanticValueBindingV1::IndexWitness {
            id: raw,
            index_space: actual,
            disjoint: false,
            availability: None,
        } = input
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_tiled_2d receiver is not thread-index authority",
            ));
        };
        let expected = SemanticDisjointIndexSpaceV1::Tiled2dIndex1d {
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
        };
        if actual != input_space
            || input_space != SemanticDisjointIndexSpaceV1::Index1d
            || output_space != expected
            || !tiled_2d_geometry_valid(lanes_per_tile, tile_rows, tile_columns, elements_per_lane)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_tiled_2d mapping identity is malformed",
            ));
        }
        let (present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Constant(Constant::Bool(true)),
            )?
            .value()
            .expect("emitted tiled-2d witness predicate");
        Ok(SemanticValueBindingV1::OptionComponentWitness {
            present,
            raw,
            index_space: expected,
            availability: self
                .option_dominance
                .availability(
                    call.destination()
                        .expect("checked destination")
                        .place()
                        .local(),
                )
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "checked-tiled-2d Option lacks an authenticated Some edge",
                    )
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_checked_row_striped_2d(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        input_space: SemanticDisjointIndexSpaceV1,
        output_space: SemanticDisjointIndexSpaceV1,
        lanes_per_row: u64,
        elements_per_lane: u64,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 1)?;
        let input = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        let SemanticValueBindingV1::IndexWitness {
            id: raw,
            index_space: actual,
            disjoint: false,
            availability: None,
        } = input
        else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_row_striped_2d receiver is not thread-index authority",
            ));
        };
        let expected = SemanticDisjointIndexSpaceV1::RowStriped2dIndex1d {
            lanes_per_row,
            elements_per_lane,
        };
        if actual != input_space
            || input_space != SemanticDisjointIndexSpaceV1::Index1d
            || output_space != expected
            || !row_striped_2d_geometry_valid(lanes_per_row, elements_per_lane)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked_row_striped_2d mapping identity is malformed",
            ));
        }
        let (present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Constant(Constant::Bool(true)),
            )?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        Ok(SemanticValueBindingV1::OptionComponentWitness {
            present,
            raw,
            index_space: expected,
            availability: self
                .option_dominance
                .availability(
                    call.destination()
                        .ok_or_else(|| {
                            unsupported(
                                0,
                                Some(block.index()),
                                None,
                                "checked row-striped destination is missing",
                            )
                        })?
                        .place()
                        .local(),
                )
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "checked-row-striped-2d Option lacks an authenticated Some edge",
                    )
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_block_component_index(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
        raw: ValueId,
        component: ValueId,
        lanes_per_block: u64,
        elements_per_lane: u64,
    ) -> Result<(ValueId, ValueId), ProductionSemanticKirErrorV1> {
        let Some(block_elements) = lanes_per_block.checked_mul(elements_per_lane) else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "blocked dimensions overflow during component projection",
            ));
        };
        if lanes_per_block == 0 || elements_per_lane == 0 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "blocked dimensions are zero during component projection",
            ));
        }
        let mut constant = |value| {
            self.emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(value)),
            )
        };
        let (lanes, _) = constant(lanes_per_block)?
            .value()
            .expect("emitted lanes constant");
        let (elements, _) = constant(elements_per_lane)?
            .value()
            .expect("emitted elements constant");
        let (block_elements_value, _) = constant(block_elements)?
            .value()
            .expect("emitted block-elements constant");
        let (component_present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThan,
                    lhs: component,
                    rhs: elements,
                },
            )?
            .value()
            .expect("emitted component predicate");
        let (block_index, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    lhs: raw,
                    rhs: lanes,
                },
            )?
            .value()
            .expect("emitted block quotient");
        let (lane, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs: raw,
                    rhs: lanes,
                },
            )?
            .value()
            .expect("emitted lane remainder");
        let (block_base, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Multiply,
                    lhs: block_index,
                    rhs: block_elements_value,
                },
            )?
            .value()
            .expect("emitted block base");
        let (component_offset, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Multiply,
                    lhs: component,
                    rhs: lanes,
                },
            )?
            .value()
            .expect("emitted component offset");
        let (offset, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: component_offset,
                    rhs: lane,
                },
            )?
            .value()
            .expect("emitted blocked lane offset");
        let (index, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: block_base,
                    rhs: offset,
                },
            )?
            .value()
            .expect("emitted blocked component index");
        Ok((index, component_present))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_tiled_2d_component_index(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
        raw: ValueId,
        component: ValueId,
        rows: ValueId,
        columns: ValueId,
        row_stride: ValueId,
        lanes_per_tile: u64,
        tile_rows: u64,
        tile_columns: u64,
        elements_per_lane: u64,
    ) -> Result<(ValueId, ValueId), ProductionSemanticKirErrorV1> {
        if !tiled_2d_geometry_valid(lanes_per_tile, tile_rows, tile_columns, elements_per_lane) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "tiled-2d geometry is malformed",
            ));
        }
        let zero = self.emit_index_constant(operations, 0)?;
        let one = self.emit_index_constant(operations, 1)?;
        let maximum = self.emit_index_constant(operations, u64::MAX)?;
        let lanes = self.emit_index_constant(operations, lanes_per_tile)?;
        let tile_rows_value = self.emit_index_constant(operations, tile_rows)?;
        let tile_columns_value = self.emit_index_constant(operations, tile_columns)?;
        let elements = self.emit_index_constant(operations, elements_per_lane)?;
        let column_padding = self.emit_index_constant(operations, tile_columns - 1)?;
        let maximum_columns =
            self.emit_index_constant(operations, u64::MAX - (tile_columns - 1))?;
        let columns_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            columns,
            maximum_columns,
        )?;
        let adjusted_columns =
            self.emit_index_binary(operations, BinaryOp::Add, columns, column_padding)?;
        let tiles_per_row = self.emit_index_binary(
            operations,
            BinaryOp::Divide,
            adjusted_columns,
            tile_columns_value,
        )?;
        let tiles_nonzero =
            self.emit_compare(operations, ComparePredicate::LessThan, zero, tiles_per_row)?;
        let safe_tiles_per_row =
            self.emit_select_index(operations, tiles_nonzero, tiles_per_row, one)?;
        let tile = self.emit_index_binary(operations, BinaryOp::Divide, raw, lanes)?;
        let lane = self.emit_index_binary(operations, BinaryOp::Remainder, raw, lanes)?;
        let tile_row =
            self.emit_index_binary(operations, BinaryOp::Divide, tile, safe_tiles_per_row)?;
        let tile_column =
            self.emit_index_binary(operations, BinaryOp::Remainder, tile, safe_tiles_per_row)?;
        let lane_row =
            self.emit_index_binary(operations, BinaryOp::Divide, lane, tile_columns_value)?;
        let local_row_base =
            self.emit_index_binary(operations, BinaryOp::Multiply, lane_row, elements)?;
        let local_row =
            self.emit_index_binary(operations, BinaryOp::Add, local_row_base, component)?;
        let local_row_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            local_row_base,
            local_row,
        )?;
        let local_column =
            self.emit_index_binary(operations, BinaryOp::Remainder, lane, tile_columns_value)?;

        let maximum_tile_row =
            self.emit_index_binary(operations, BinaryOp::Divide, maximum, tile_rows_value)?;
        let tile_row_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            tile_row,
            maximum_tile_row,
        )?;
        let row_base =
            self.emit_index_binary(operations, BinaryOp::Multiply, tile_row, tile_rows_value)?;
        let row = self.emit_index_binary(operations, BinaryOp::Add, row_base, local_row)?;
        let row_add_safe =
            self.emit_compare(operations, ComparePredicate::LessThanOrEqual, row_base, row)?;

        let maximum_tile_column =
            self.emit_index_binary(operations, BinaryOp::Divide, maximum, tile_columns_value)?;
        let tile_column_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            tile_column,
            maximum_tile_column,
        )?;
        let column_base = self.emit_index_binary(
            operations,
            BinaryOp::Multiply,
            tile_column,
            tile_columns_value,
        )?;
        let column =
            self.emit_index_binary(operations, BinaryOp::Add, column_base, local_column)?;
        let column_add_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            column_base,
            column,
        )?;

        let stride_nonzero =
            self.emit_compare(operations, ComparePredicate::LessThan, zero, row_stride)?;
        let safe_stride = self.emit_select_index(operations, stride_nonzero, row_stride, one)?;
        let maximum_row =
            self.emit_index_binary(operations, BinaryOp::Divide, maximum, safe_stride)?;
        let row_multiply_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            row,
            maximum_row,
        )?;
        let row_offset = self.emit_index_binary(operations, BinaryOp::Multiply, row, row_stride)?;
        let index = self.emit_index_binary(operations, BinaryOp::Add, row_offset, column)?;
        let index_add_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            row_offset,
            index,
        )?;
        let component_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, component, elements)?;
        let stride_valid = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            columns,
            row_stride,
        )?;
        let row_valid = self.emit_compare(operations, ComparePredicate::LessThan, row, rows)?;
        let column_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, column, columns)?;
        let predicates = [
            columns_safe,
            tiles_nonzero,
            local_row_safe,
            tile_row_safe,
            row_add_safe,
            tile_column_safe,
            column_add_safe,
            row_multiply_safe,
            index_add_safe,
            component_valid,
            stride_valid,
            row_valid,
            column_valid,
        ];
        let mut present = predicates[0];
        for predicate in &predicates[1..] {
            present = self.emit_bool_and(operations, present, *predicate)?;
        }
        Ok((index, present))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_row_striped_2d_component_index(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
        raw: ValueId,
        component: ValueId,
        rows: ValueId,
        columns: ValueId,
        row_stride: ValueId,
        lanes_per_row: u64,
        elements_per_lane: u64,
    ) -> Result<(ValueId, ValueId), ProductionSemanticKirErrorV1> {
        if !row_striped_2d_geometry_valid(lanes_per_row, elements_per_lane) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "row-striped-2d geometry is malformed",
            ));
        }
        let zero = self.emit_index_constant(operations, 0)?;
        let one = self.emit_index_constant(operations, 1)?;
        let maximum = self.emit_index_constant(operations, u64::MAX)?;
        let lanes = self.emit_index_constant(operations, lanes_per_row)?;
        let elements = self.emit_index_constant(operations, elements_per_lane)?;

        let row = self.emit_index_binary(operations, BinaryOp::Divide, raw, lanes)?;
        let lane = self.emit_index_binary(operations, BinaryOp::Remainder, raw, lanes)?;
        let maximum_component =
            self.emit_index_binary(operations, BinaryOp::Divide, maximum, lanes)?;
        let component_multiply_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            component,
            maximum_component,
        )?;
        let column_base =
            self.emit_index_binary(operations, BinaryOp::Multiply, component, lanes)?;
        let column = self.emit_index_binary(operations, BinaryOp::Add, column_base, lane)?;
        let column_add_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            column_base,
            column,
        )?;

        let stride_nonzero =
            self.emit_compare(operations, ComparePredicate::LessThan, zero, row_stride)?;
        let safe_stride = self.emit_select_index(operations, stride_nonzero, row_stride, one)?;
        let maximum_row =
            self.emit_index_binary(operations, BinaryOp::Divide, maximum, safe_stride)?;
        let row_multiply_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            row,
            maximum_row,
        )?;
        let row_offset = self.emit_index_binary(operations, BinaryOp::Multiply, row, row_stride)?;
        let index = self.emit_index_binary(operations, BinaryOp::Add, row_offset, column)?;
        let index_add_safe = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            row_offset,
            index,
        )?;

        let component_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, component, elements)?;
        let stride_valid = self.emit_compare(
            operations,
            ComparePredicate::LessThanOrEqual,
            columns,
            row_stride,
        )?;
        let row_valid = self.emit_compare(operations, ComparePredicate::LessThan, row, rows)?;
        let column_valid =
            self.emit_compare(operations, ComparePredicate::LessThan, column, columns)?;
        let predicates = [
            component_multiply_safe,
            column_add_safe,
            row_multiply_safe,
            index_add_safe,
            component_valid,
            stride_valid,
            row_valid,
            column_valid,
        ];
        let mut present = predicates[0];
        for predicate in &predicates[1..] {
            present = self.emit_bool_and(operations, present, *predicate)?;
        }
        Ok((index, present))
    }

    fn coerce_index(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
        binding: SemanticValueBindingV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let (id, ty) = binding
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if ty == Type::INDEX {
            return Ok(SemanticValueBindingV1::Value { id, ty });
        }
        if ty != Type::Scalar(ScalarType::U64) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "exclusive access index is not usize",
            ));
        }
        self.emit(
            operations,
            Type::INDEX,
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value: id,
                to: Type::INDEX,
            },
        )
    }

    fn lower_checked_slice_access(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        receiver: usize,
        index: SemanticValueBindingV1,
        precondition: Option<ValueId>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let (slice, slice_ty) = self
            .lower_operand(block, None, &call.arguments()[receiver], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let (index, index_ty) = index
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let Type::Slice(slice_type) = slice_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked DisjointSlice receiver is not a lowered slice",
            ));
        };
        if index_ty != Type::INDEX {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "checked DisjointSlice index is not a trusted index",
            ));
        }
        let (length, _) = self
            .emit(
                operations,
                Type::INDEX,
                OperationKind::SliceLength { slice },
            )?
            .value()
            .expect("emitted scalar value");
        let (extent_present, _) = self
            .emit(
                operations,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThan,
                    lhs: index,
                    rhs: length,
                },
            )?
            .value()
            .expect("emitted scalar value");
        let present = if let Some(precondition) = precondition {
            self.emit(
                operations,
                Type::BOOL,
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: precondition,
                    rhs: extent_present,
                },
            )?
            .value()
            .expect("emitted combined checked-access predicate")
            .0
        } else {
            extent_present
        };
        let pointer_ty = Type::pointer(
            (*slice_type.element).clone(),
            slice_type.address_space,
            slice_type.access,
        );
        let (base, _) = self
            .emit(
                operations,
                pointer_ty.clone(),
                OperationKind::SliceData { slice },
            )?
            .value()
            .expect("emitted scalar value");
        let (pointer, _) = self
            .emit(
                operations,
                pointer_ty.clone(),
                OperationKind::GetElementPointer {
                    base,
                    offset: index,
                },
            )?
            .value()
            .expect("emitted scalar value");
        Ok(SemanticValueBindingV1::OptionPointer {
            present,
            pointer,
            pointer_ty,
        })
    }

    fn emit_index_intrinsic(
        &mut self,
        operations: &mut Vec<Operation>,
        kind: IndexKind,
        axis: Axis,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.emit(
            operations,
            Type::INDEX,
            OperationKind::Intrinsic(IntrinsicOperation::new(
                IntrinsicKind::InvocationIndex { kind, axis },
                Type::INDEX,
            )),
        )
    }

    fn emit_launch_index_v1(
        &mut self,
        operations: &mut Vec<Operation>,
        kind: IndexKind,
        axis: Axis,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        match inactive_launch_axis_value_v1(self.launch_rank, kind, axis) {
            Some(value) => self.emit(
                operations,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(value)),
            ),
            None => self.emit_index_intrinsic(operations, kind, axis),
        }
    }

    fn assign_place(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        destination: &SemanticPlaceV1,
        value: SemanticValueBindingV1,
        volatility: SemanticVolatilityV1,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if destination.projections().is_empty() {
            self.store_enum_payload_v1(block, statement, destination.local(), &value, operations)?;
            return self.bind_destination(block, statement, destination, value);
        }
        if !destination
            .projections()
            .iter()
            .any(|projection| projection.kind() == SemanticProjectionKindV1::Dereference)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "projected local assignment is not a dereferenced store",
            ));
        }
        let (pointer, pointer_ty) = self
            .resolve_place(block, statement, destination)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
        let (value, value_ty) = value
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
        let Type::Pointer(pointer_type) = pointer_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "dereferenced store destination is not a lowered pointer",
            ));
        };
        if *pointer_type.pointee != value_ty {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "dereferenced store value type differs from its pointee",
            ));
        }
        let mut access =
            memory_access_for_type(self.types, destination.ty(), pointer_type.address_space)?;
        access.volatile = volatility == SemanticVolatilityV1::Volatile;
        self.push_operation(operations, || {
            Operation::new(
                vec![],
                OperationKind::Store {
                    pointer,
                    value,
                    access,
                },
            )
        })?;
        Ok(())
    }

    fn store_enum_payload_v1(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        local: SemanticLocalIdV1,
        value: &SemanticValueBindingV1,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let SemanticValueBindingV1::Enum {
            variant: Some(variant),
            payloads,
            ..
        } = value
        else {
            return Ok(());
        };
        let Some(fields) = payloads.get(variant) else {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "variant-refined enum has no retained payload",
            ));
        };
        for (field, binding) in fields.iter().enumerate() {
            let key = (local.index(), *variant, field as u32);
            if self.enum_payload_sources.contains_key(&key)
                && semantic_binding_can_restore_from_unique_source_v1(binding)
            {
                if self
                    .enum_payload_compile_time_custody
                    .insert(
                        key,
                        SemanticEnumPayloadCustodyV1 {
                            source_block: block,
                            binding: binding.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "enum payload compile-time custody was assigned more than once",
                    ));
                }
                if !self.enum_payload_storage.contains_key(&key) || binding.values().is_err() {
                    continue;
                }
            }
            if self
                .enum_payload_requires_compile_time_custody
                .contains(&key)
            {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    statement,
                    "non-storable enum payload cannot be retained from its unique source",
                ));
            }
            let Some(storage) = self.enum_payload_storage.get(&key).cloned() else {
                continue;
            };
            let values = match storage.exact_enum_variant {
                Some(expected_variant) => {
                    exact_enum_binding_values_v1(binding, expected_variant)
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?
                }
                None => binding
                    .values()
                    .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?,
            };
            if values.len() != storage.components.len() {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    statement,
                    "enum payload changed its private-storage component types",
                ));
            }
            for ((value, actual), component) in values.into_iter().zip(storage.components.iter()) {
                let value = self.coerce_transport_value_v1(
                    operations,
                    block,
                    statement,
                    value,
                    actual,
                    component.kernel_type.clone(),
                    "enum payload changed its private-storage component types",
                )?;
                self.push_operation(operations, || {
                    Operation::new(
                        vec![],
                        OperationKind::Store {
                            pointer: component.pointer,
                            value,
                            access: MemoryAccess::new(AddressSpace::Private, component.alignment),
                        },
                    )
                })?;
            }
        }
        Ok(())
    }

    fn lower_indexed_place_address(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        place: &SemanticPlaceV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let local = self.require_local(block, statement, place.local().index())?;
        let mut binding = self.locals[local].clone().ok_or(
            ProductionSemanticKirErrorV1::MissingLocalDefinition {
                function: 0,
                block: block.index(),
                statement,
                local: place.local().index(),
            },
        )?;
        for projection in place.projections() {
            match projection.kind() {
                SemanticProjectionKindV1::Dereference
                | SemanticProjectionKindV1::Field(0)
                | SemanticProjectionKindV1::Downcast(_)
                | SemanticProjectionKindV1::OpaqueCast
                | SemanticProjectionKindV1::Subtype => {}
                SemanticProjectionKindV1::Index(index_local) => {
                    let (slice, slice_ty) = binding
                        .value()
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                    let Type::Slice(slice_type) = slice_ty else {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "indexed semantic place is not a lowered slice",
                        ));
                    };
                    let index_binding = self
                        .locals
                        .get(index_local.index() as usize)
                        .and_then(Option::as_ref)
                        .ok_or(ProductionSemanticKirErrorV1::MissingLocalDefinition {
                            function: 0,
                            block: block.index(),
                            statement,
                            local: index_local.index(),
                        })?;
                    let (index, index_ty) = index_binding
                        .value()
                        .map_err(|detail| unsupported(0, Some(block.index()), statement, detail))?;
                    let index = if index_ty == Type::INDEX {
                        index
                    } else if index_ty == Type::Scalar(ScalarType::U64) {
                        self.emit(
                            operations,
                            Type::INDEX,
                            OperationKind::Cast {
                                kind: CastKind::Bitcast,
                                value: index,
                                to: Type::INDEX,
                            },
                        )?
                        .value()
                        .expect("emitted index cast")
                        .0
                    } else {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "slice index has no exact Kernel IR index representation",
                        ));
                    };
                    let pointer_ty = Type::pointer(
                        (*slice_type.element).clone(),
                        slice_type.address_space,
                        slice_type.access,
                    );
                    let base = self
                        .emit(
                            operations,
                            pointer_ty.clone(),
                            OperationKind::SliceData { slice },
                        )?
                        .value()
                        .expect("emitted slice data")
                        .0;
                    binding = self.emit(
                        operations,
                        pointer_ty,
                        OperationKind::GetElementPointer {
                            base,
                            offset: index,
                        },
                    )?;
                }
                SemanticProjectionKindV1::ConstantIndex { .. }
                | SemanticProjectionKindV1::Subslice { .. }
                | SemanticProjectionKindV1::Field(_) => {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "indexed semantic place has an unsupported projection",
                    ));
                }
            }
        }
        Ok(binding)
    }

    fn bind_destination(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        destination: &SemanticPlaceV1,
        value: SemanticValueBindingV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if !destination.projections().is_empty() {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic result destination is projected",
            ));
        }
        let index = self.require_local(block, statement, destination.local().index())?;
        self.locals[index] = Some(value);
        Ok(())
    }

    fn resolve_place(
        &self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        place: &SemanticPlaceV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let index = self.require_local(block, statement, place.local().index())?;
        let mut binding = self.locals[index].clone().ok_or(
            ProductionSemanticKirErrorV1::MissingLocalDefinition {
                function: 0,
                block: block.index(),
                statement,
                local: place.local().index(),
            },
        )?;
        for projection in place.projections() {
            binding = match (binding, projection.kind()) {
                (SemanticValueBindingV1::Unit, _) => {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "unit capability result cannot be projected",
                    ));
                }
                (SemanticValueBindingV1::Unmaterialized, _) => {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "unmaterialized enum payload cannot be observed",
                    ));
                }
                (SemanticValueBindingV1::MatrixContext, SemanticProjectionKindV1::Dereference)
                | (
                    SemanticValueBindingV1::MatrixContext,
                    SemanticProjectionKindV1::OpaqueCast | SemanticProjectionKindV1::Subtype,
                ) => SemanticValueBindingV1::MatrixContext,
                (
                    SemanticValueBindingV1::MathContext,
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => SemanticValueBindingV1::MathContext,
                (
                    SemanticValueBindingV1::CollectiveContext,
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => SemanticValueBindingV1::CollectiveContext,
                (
                    SemanticValueBindingV1::Aggregate(fields),
                    SemanticProjectionKindV1::Field(field),
                ) => fields.get(field as usize).cloned().ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block.index()),
                        statement,
                        "aggregate field projection is out of range",
                    )
                })?,
                (
                    SemanticValueBindingV1::Enum {
                        discriminant,
                        discriminant_ty,
                        semantic_type,
                        variant,
                        payloads,
                    },
                    SemanticProjectionKindV1::Downcast(expected),
                ) => {
                    if variant.is_some_and(|variant| variant != expected) {
                        return Err(unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "enum downcast does not match its known variant",
                        ));
                    }
                    SemanticValueBindingV1::Enum {
                        discriminant,
                        discriminant_ty,
                        semantic_type,
                        variant: Some(expected),
                        payloads,
                    }
                }
                (
                    SemanticValueBindingV1::Enum {
                        variant: Some(variant),
                        payloads,
                        ..
                    },
                    SemanticProjectionKindV1::Field(field),
                ) => {
                    let available_fields = payloads.get(&variant).map_or(0, Vec::len);
                    let projected = project_enum_payload_field(variant, &payloads, field);
                    if matches!(
                        projected,
                        Ok(SemanticValueBindingV1::Unmaterialized) | Err(_)
                    ) {
                        return Err(ProductionSemanticKirErrorV1::EnumPayloadUnavailable {
                            function: 0,
                            block: block.index(),
                            statement,
                            local: place.local().index(),
                            variant,
                            field,
                            available_fields,
                            evidence: self.enum_carrier_evidence_v1(place.local()),
                        });
                    }
                    projected.expect("available enum field was checked")
                }
                (
                    binding @ SemanticValueBindingV1::Enum { .. },
                    SemanticProjectionKindV1::OpaqueCast | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    SemanticValueBindingV1::Aggregate(fields),
                    SemanticProjectionKindV1::ConstantIndex {
                        offset, from_end, ..
                    },
                ) => {
                    let offset = usize::try_from(offset).map_err(|_| {
                        unsupported(
                            0,
                            Some(block.index()),
                            statement,
                            "aggregate constant index does not fit this host",
                        )
                    })?;
                    let index = if from_end {
                        fields.len().checked_sub(offset)
                    } else {
                        Some(offset)
                    };
                    index
                        .and_then(|index| fields.get(index))
                        .cloned()
                        .ok_or_else(|| {
                            unsupported(
                                0,
                                Some(block.index()),
                                statement,
                                "aggregate constant index is out of range",
                            )
                        })?
                }
                (
                    binding @ SemanticValueBindingV1::Aggregate(_),
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::WorkgroupPipeline { .. },
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    SemanticValueBindingV1::OptionPointer {
                        pointer,
                        pointer_ty,
                        ..
                    },
                    SemanticProjectionKindV1::Field(_),
                ) => SemanticValueBindingV1::Value {
                    id: pointer,
                    ty: pointer_ty,
                },
                (
                    SemanticValueBindingV1::OptionIndexWitness {
                        id,
                        index_space,
                        availability,
                        ..
                    },
                    SemanticProjectionKindV1::Field(_),
                ) => SemanticValueBindingV1::IndexWitness {
                    id,
                    availability: Some(SemanticCapabilityAvailabilityV1::Option(availability)),
                    index_space,
                    disjoint: true,
                },
                (
                    SemanticValueBindingV1::OptionComponentWitness {
                        raw,
                        index_space,
                        availability,
                        ..
                    },
                    SemanticProjectionKindV1::Field(_),
                ) => SemanticValueBindingV1::ComponentWitness {
                    raw,
                    index_space,
                    availability: SemanticCapabilityAvailabilityV1::Option(availability),
                },
                (
                    SemanticValueBindingV1::OptionGridLeader { availability, .. },
                    SemanticProjectionKindV1::Field(_),
                ) => SemanticValueBindingV1::GridLeader {
                    availability: SemanticCapabilityAvailabilityV1::Option(availability),
                },
                (
                    binding @ SemanticValueBindingV1::OptionPointer { .. },
                    SemanticProjectionKindV1::Downcast(_),
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::OptionIndexWitness { .. },
                    SemanticProjectionKindV1::Downcast(_),
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::OptionGridLeader { .. },
                    SemanticProjectionKindV1::Downcast(_),
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::OptionComponentWitness { .. },
                    SemanticProjectionKindV1::Downcast(_),
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::Value { .. },
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::Field(0)
                    | SemanticProjectionKindV1::Downcast(_)
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::IndexWitness { .. },
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::Field(0)
                    | SemanticProjectionKindV1::Downcast(_)
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::GridLeader { .. },
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::Field(0)
                    | SemanticProjectionKindV1::Downcast(_)
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (
                    binding @ SemanticValueBindingV1::ComponentWitness { .. },
                    SemanticProjectionKindV1::Dereference
                    | SemanticProjectionKindV1::Field(0)
                    | SemanticProjectionKindV1::Downcast(_)
                    | SemanticProjectionKindV1::OpaqueCast
                    | SemanticProjectionKindV1::Subtype,
                ) => binding,
                (binding, projection) => {
                    let mut evidence = self.enum_carrier_evidence_v1(place.local());
                    if let SemanticProjectionKindV1::Index(index) = projection {
                        evidence.extend(
                            self.enum_carrier_evidence_v1(index)
                                .into_iter()
                                .map(|record| format!("index {record}")),
                        );
                        evidence.truncate(16);
                    }
                    return Err(ProductionSemanticKirErrorV1::PlaceProjectionUnavailable {
                        function: 0,
                        block: block.index(),
                        statement,
                        local: place.local().index(),
                        binding: semantic_binding_kind_v1(&binding),
                        projection: format!("{projection:?}"),
                        evidence,
                    });
                }
            };
        }
        let availability = match &binding {
            SemanticValueBindingV1::IndexWitness {
                availability: Some(availability),
                ..
            } => Some(*availability),
            SemanticValueBindingV1::GridLeader { availability } => Some(*availability),
            SemanticValueBindingV1::ComponentWitness { availability, .. } => Some(*availability),
            SemanticValueBindingV1::Unit
            | SemanticValueBindingV1::Unmaterialized
            | SemanticValueBindingV1::Aggregate(_)
            | SemanticValueBindingV1::Enum { .. }
            | SemanticValueBindingV1::MathContext
            | SemanticValueBindingV1::CollectiveContext
            | SemanticValueBindingV1::WorkgroupLdsScope
            | SemanticValueBindingV1::MatrixContext
            | SemanticValueBindingV1::WaveLane { .. }
            | SemanticValueBindingV1::MatrixFragment { .. }
            | SemanticValueBindingV1::AccumulatorFragment { .. }
            | SemanticValueBindingV1::Gfx950LdsTransposeTile { .. }
            | SemanticValueBindingV1::WorkgroupPipeline { .. }
            | SemanticValueBindingV1::Value { .. }
            | SemanticValueBindingV1::OptionPointer { .. }
            | SemanticValueBindingV1::IndexWitness {
                availability: None, ..
            }
            | SemanticValueBindingV1::OptionIndexWitness { .. }
            | SemanticValueBindingV1::OptionComponentWitness { .. }
            | SemanticValueBindingV1::OptionGridLeader { .. } => None,
        };
        if availability.is_some_and(|availability| match availability {
            SemanticCapabilityAvailabilityV1::Option(availability) => {
                !self.option_dominance.allows(availability, block)
            }
            SemanticCapabilityAvailabilityV1::EnumPayload { local, variant } => {
                !self.enum_variant_is_available_v1(local, variant, block)
            }
        }) {
            return Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "capability payload is used outside its authenticated enum edge",
            ));
        }
        Ok(binding)
    }

    fn enum_carrier_evidence_v1(&self, local: SemanticLocalIdV1) -> Vec<String> {
        const MAX_EVIDENCE_V1: usize = 16;

        let mut evidence = Vec::new();
        let mut pending = VecDeque::from([local]);
        let mut visited = BTreeSet::new();
        while let Some(local) = pending.pop_front() {
            if !visited.insert(local.index()) {
                continue;
            }
            if let Some(declaration) = self.function.locals().get(local.index() as usize) {
                let shape = self
                    .types
                    .get(declaration.ty().index() as usize)
                    .map(|declaration| format!("{:?}", declaration.shape()))
                    .unwrap_or_else(|| "<missing type declaration>".to_owned());
                evidence.push(format!(
                    "local {} declaration: role {:?}, type {}, shape {shape}",
                    local.index(),
                    declaration.role(),
                    declaration.ty().index(),
                ));
                if evidence.len() == MAX_EVIDENCE_V1 {
                    return evidence;
                }
            }
            for (block_index, block) in self.function.blocks().iter().enumerate() {
                for (statement_index, statement) in block.statements().iter().enumerate() {
                    let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                        continue;
                    };
                    if assignment.destination().projections().is_empty()
                        && assignment.destination().local() == local
                    {
                        evidence.push(format!(
                            "local {} definition block {block_index} statement {statement_index}: {:?}",
                            local.index(),
                            assignment.value().kind(),
                        ));
                        let _: Result<(), ()> =
                            assignment.value().kind().try_visit_operands(|operand| {
                                if let SemanticOperandV1::Copy(place)
                                | SemanticOperandV1::Move(place) = operand
                                {
                                    pending.push_back(place.local());
                                }
                                Ok(())
                            });
                        match assignment.value().kind() {
                            SemanticRvalueKindV1::Borrow { place, .. }
                            | SemanticRvalueKindV1::AddressOf { place, .. }
                            | SemanticRvalueKindV1::Length(place)
                            | SemanticRvalueKindV1::Discriminant(place) => {
                                pending.push_back(place.local());
                            }
                            SemanticRvalueKindV1::Load(load) => {
                                pending.push_back(load.source().local());
                            }
                            SemanticRvalueKindV1::Use(_)
                            | SemanticRvalueKindV1::Unary { .. }
                            | SemanticRvalueKindV1::Binary { .. }
                            | SemanticRvalueKindV1::CheckedBinary(_)
                            | SemanticRvalueKindV1::UncheckedBinary(_)
                            | SemanticRvalueKindV1::Cast { .. }
                            | SemanticRvalueKindV1::Aggregate(_) => {}
                        }
                    }
                    if matches!(
                        assignment.value().kind(),
                        SemanticRvalueKindV1::Discriminant(place)
                            if place.projections().is_empty() && place.local() == local
                    ) {
                        evidence.push(format!(
                            "local {} discriminant block {block_index} statement {statement_index}: destination local {}; terminator {:?}",
                            local.index(),
                            assignment.destination().local().index(),
                            block.terminator().kind(),
                        ));
                    }
                    if evidence.len() == MAX_EVIDENCE_V1 {
                        return evidence;
                    }
                }
            }
        }
        evidence
    }

    fn emit(
        &mut self,
        operations: &mut Vec<Operation>,
        ty: Type,
        kind: OperationKind,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let id = ValueId(self.next_value);
        let emitted_u32_constant = if ty == Type::Scalar(ScalarType::U32) {
            match &kind {
                OperationKind::Constant(Constant::U32(constant)) => Some(*constant),
                _ => None,
            }
        } else {
            None
        };
        let emitted_u32_bitand_mask = if ty == Type::Scalar(ScalarType::U32) {
            match &kind {
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs,
                    rhs,
                } => [*lhs, *rhs]
                    .into_iter()
                    .filter_map(|operand| self.emitted_u32_constants.get(&operand).copied())
                    .min(),
                _ => None,
            }
        } else {
            None
        };
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or_else(|| unsupported(0, None, None, "Kernel IR SSA identity overflow"))?;
        self.push_operation(operations, || {
            Operation::effect_free(ValueDef::new(id, ty.clone()), kind)
        })?;
        if let Some(constant) = emitted_u32_constant {
            self.emitted_u32_constants.insert(id, constant);
        }
        if let Some(mask) = emitted_u32_bitand_mask {
            self.emitted_u32_bitand_masks.insert(id, mask);
        }
        Ok(SemanticValueBindingV1::Value { id, ty })
    }

    fn lower_workgroup_reduce_sum(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operations: &mut Vec<Operation>,
        element: SemanticTypeIdV1,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 4)?;
        let workgroup = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        if !matches!(workgroup, SemanticValueBindingV1::Aggregate(_)) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction lacks the compiler-derived workgroup snapshot",
            ));
        }
        let context = self.lower_operand(block, None, &call.arguments()[1], operations)?;
        if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction lacks compiler-issued collective authority",
            ));
        }
        let scratch = self.lower_operand(block, None, &call.arguments()[2], operations)?;
        let SemanticValueBindingV1::Aggregate(scratch_fields) = scratch else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction scratch is not the reviewed aggregate representation",
            ));
        };
        if scratch_fields.len() != 4 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction scratch field count changed",
            ));
        }
        let (scratch, scratch_ty) = scratch_fields[0]
            .clone()
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let (_, slots_ty) = scratch_fields[1]
            .clone()
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let scalar = lower_scalar_type(self.types, element)?;
        if !matches!(
            scalar,
            Type::Scalar(ScalarType::U32 | ScalarType::I32 | ScalarType::F32)
        ) || slots_ty != Type::Scalar(ScalarType::U32)
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction requires matching u32, i32, or f32 scratch",
            ));
        }
        let Type::Pointer(scratch_pointer) = scratch_ty else {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction scratch base is not a lowered pointer",
            ));
        };
        if scratch_pointer.address_space != AddressSpace::Workgroup
            || scratch_pointer.access != AccessMode::ReadWrite
            || *scratch_pointer.pointee != scalar
        {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction scratch did not originate as matching compiler-owned LDS",
            ));
        }
        let (value, value_ty) = self
            .lower_operand(block, None, &call.arguments()[3], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if value_ty != scalar {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction input differs from its scratch element type",
            ));
        }
        let [size, y, z] = self.required_workgroup.ok_or_else(|| {
            unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction requires an exact source launch contract",
            )
        })?;
        if y != 1 || z != 1 || size == 0 || !size.is_power_of_two() || size > 256 {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "workgroup reduction requires a one-dimensional power-of-two workgroup no larger than 256",
            ));
        }

        let rank = self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Intrinsic(IntrinsicOperation::new(
                IntrinsicKind::InvocationIndex {
                    kind: IndexKind::Local,
                    axis: Axis::X,
                },
                Type::INDEX,
            )),
        )?;
        self.emit_workgroup_store_at(operations, scratch, rank, value, &scalar)?;
        self.emit_collective_barrier(operations)?;

        let mut offset = size >> 1;
        while offset != 0 {
            let offset_value = self.emit_index_constant(operations, u64::from(offset))?;
            let active =
                self.emit_compare(operations, ComparePredicate::LessThan, rank, offset_value)?;
            let pair = self.emit_index_binary(operations, BinaryOp::Add, rank, offset_value)?;
            let zero = self.emit_index_constant(operations, 0)?;
            let safe_pair = self.emit_select_index(operations, active, pair, zero)?;
            let lhs = self.emit_workgroup_load_at(operations, scratch, rank, &scalar)?;
            let rhs = self.emit_workgroup_load_at(operations, scratch, safe_pair, &scalar)?;
            let sum = self.emit_id(
                operations,
                scalar.clone(),
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs,
                    rhs,
                },
            )?;
            let next = self.emit_id(
                operations,
                scalar.clone(),
                OperationKind::Select {
                    condition: active,
                    true_value: sum,
                    false_value: lhs,
                },
            )?;
            self.emit_collective_barrier(operations)?;
            self.emit_workgroup_store_at(operations, scratch, rank, next, &scalar)?;
            self.emit_collective_barrier(operations)?;
            offset >>= 1;
        }
        let zero = self.emit_index_constant(operations, 0)?;
        let result = self.emit_workgroup_load_at(operations, scratch, zero, &scalar)?;
        self.emit_collective_barrier(operations)?;
        Ok(SemanticValueBindingV1::Value {
            id: result,
            ty: scalar,
        })
    }

    fn emit_workgroup_pointer_at(
        &mut self,
        operations: &mut Vec<Operation>,
        base: ValueId,
        offset: ValueId,
        element: &Type,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::pointer(
                element.clone(),
                AddressSpace::Workgroup,
                AccessMode::ReadWrite,
            ),
            OperationKind::GetElementPointer { base, offset },
        )
    }

    fn emit_workgroup_load_at(
        &mut self,
        operations: &mut Vec<Operation>,
        base: ValueId,
        offset: ValueId,
        element: &Type,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        let pointer = self.emit_workgroup_pointer_at(operations, base, offset, element)?;
        self.emit_id(
            operations,
            element.clone(),
            OperationKind::Load {
                pointer,
                access: MemoryAccess::new(AddressSpace::Workgroup, 4),
            },
        )
    }

    fn emit_workgroup_store_at(
        &mut self,
        operations: &mut Vec<Operation>,
        base: ValueId,
        offset: ValueId,
        value: ValueId,
        element: &Type,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let pointer = self.emit_workgroup_pointer_at(operations, base, offset, element)?;
        self.push_operation(operations, || {
            Operation::new(
                Vec::new(),
                OperationKind::Store {
                    pointer,
                    value,
                    access: MemoryAccess::new(AddressSpace::Workgroup, 4),
                },
            )
        })
    }

    fn emit_collective_barrier(
        &mut self,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.push_operation(operations, || {
            Operation::new(
                Vec::new(),
                OperationKind::WorkgroupBarrier(WorkgroupBarrier {
                    memory_scope: SynchronizationScope::Workgroup,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::AcquireRelease,
                        [AddressSpace::Workgroup],
                    ),
                    convergence: Convergence::uniform(SynchronizationScope::Workgroup),
                }),
            )
        })
    }

    fn emit_checked_binary(
        &mut self,
        operations: &mut Vec<Operation>,
        ty: Type,
        operator: CheckedBinaryOperator,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let value = ValueId(self.next_value);
        let overflow = ValueId(
            self.next_value
                .checked_add(1)
                .ok_or_else(|| unsupported(0, None, None, "Kernel IR SSA identity overflow"))?,
        );
        let next_value = self
            .next_value
            .checked_add(2)
            .ok_or_else(|| unsupported(0, None, None, "Kernel IR SSA identity overflow"))?;
        self.push_operation(operations, || {
            Operation::checked_binary(
                ValueDef::new(value, ty.clone()),
                ValueDef::new(overflow, Type::BOOL),
                operator,
                lhs,
                rhs,
            )
        })?;
        self.next_value = next_value;
        Ok(SemanticValueBindingV1::Aggregate(vec![
            SemanticValueBindingV1::Value { id: value, ty },
            SemanticValueBindingV1::Value {
                id: overflow,
                ty: Type::BOOL,
            },
        ]))
    }

    fn emit_float_operation(
        &mut self,
        operations: &mut Vec<Operation>,
        operation: FloatOperation,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let id = ValueId(self.next_value);
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or_else(|| unsupported(0, None, None, "Kernel IR SSA identity overflow"))?;
        let ty = operation.result_type();
        self.push_operation(operations, || operation.operation(id))?;
        Ok(SemanticValueBindingV1::Value { id, ty })
    }

    fn reserve_operation(
        &mut self,
        operations: &mut Vec<Operation>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let block_actual =
            operations
                .len()
                .checked_add(1)
                .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::Operations,
                    actual: usize::MAX,
                    limit: MAX_BLOCK_OPERATIONS_V1,
                })?;
        enforce_limit(
            ProductionSemanticKirResourceV1::Operations,
            block_actual,
            MAX_BLOCK_OPERATIONS_V1,
        )?;
        let total_actual = self.emitted_operations.checked_add(1).ok_or(
            ProductionSemanticKirErrorV1::ResourceLimit {
                resource: ProductionSemanticKirResourceV1::Operations,
                actual: usize::MAX,
                limit: self.max_operations,
            },
        )?;
        enforce_limit(
            ProductionSemanticKirResourceV1::Operations,
            total_actual,
            self.max_operations,
        )?;
        operations
            .try_reserve(1)
            .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
                resource: ProductionSemanticKirResourceV1::Operations,
            })?;
        self.emitted_operations = total_actual;
        Ok(())
    }

    fn push_operation(
        &mut self,
        operations: &mut Vec<Operation>,
        build: impl FnOnce() -> Operation,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.reserve_operation(operations)?;
        operations.push(build());
        Ok(())
    }

    fn emit_id(
        &mut self,
        operations: &mut Vec<Operation>,
        ty: Type,
        kind: OperationKind,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit(operations, ty, kind)?
            .value()
            .map(|(id, _)| id)
            .map_err(|detail| unsupported(0, None, None, detail))
    }

    fn coerce_transport_value_v1(
        &mut self,
        operations: &mut Vec<Operation>,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        value: ValueId,
        actual: Type,
        expected: Type,
        description: &'static str,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        if actual == expected {
            return Ok(value);
        }
        if !index_and_u64_are_transport_equivalent(&actual, &expected) {
            return Err(unsupported(0, Some(block.index()), statement, description));
        }
        self.emit_id(
            operations,
            expected.clone(),
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value,
                to: expected,
            },
        )
    }

    fn emit_index_constant(
        &mut self,
        operations: &mut Vec<Operation>,
        value: u64,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Constant(Constant::Index(value)),
        )
    }

    fn emit_index_binary(
        &mut self,
        operations: &mut Vec<Operation>,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Binary { op, lhs, rhs },
        )
    }

    fn emit_compare(
        &mut self,
        operations: &mut Vec<Operation>,
        predicate: ComparePredicate,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::BOOL,
            OperationKind::Compare {
                predicate,
                lhs,
                rhs,
            },
        )
    }

    fn emit_select_index(
        &mut self,
        operations: &mut Vec<Operation>,
        condition: ValueId,
        true_value: ValueId,
        false_value: ValueId,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::INDEX,
            OperationKind::Select {
                condition,
                true_value,
                false_value,
            },
        )
    }

    fn emit_bool_and(
        &mut self,
        operations: &mut Vec<Operation>,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::BOOL,
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs,
                rhs,
            },
        )
    }

    fn emit_bool_or(
        &mut self,
        operations: &mut Vec<Operation>,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<ValueId, ProductionSemanticKirErrorV1> {
        self.emit_id(
            operations,
            Type::BOOL,
            OperationKind::Binary {
                op: BinaryOp::BitOr,
                lhs,
                rhs,
            },
        )
    }

    fn emit_checked_index(
        &mut self,
        operations: &mut Vec<Operation>,
        operator: CheckedBinaryOperator,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(ValueId, ValueId), ProductionSemanticKirErrorV1> {
        let SemanticValueBindingV1::Aggregate(parts) =
            self.emit_checked_binary(operations, Type::INDEX, operator, lhs, rhs)?
        else {
            unreachable!("checked binary lowering returns value and overflow");
        };
        let (value, _) = parts[0]
            .value()
            .expect("checked binary value has plain representation");
        let (overflow, _) = parts[1]
            .value()
            .expect("checked binary overflow has plain representation");
        let safe = self.emit_id(
            operations,
            Type::BOOL,
            OperationKind::Unary {
                op: UnaryOp::Not,
                operand: overflow,
            },
        )?;
        Ok((value, safe))
    }

    fn emit_results(
        &mut self,
        operations: &mut Vec<Operation>,
        types: Vec<Type>,
        kind: OperationKind,
    ) -> Result<Vec<ValueDef>, ProductionSemanticKirErrorV1> {
        self.reserve_operation(operations)?;
        let mut results = Vec::with_capacity(types.len());
        for ty in types {
            let id = ValueId(self.next_value);
            self.next_value = self
                .next_value
                .checked_add(1)
                .ok_or_else(|| unsupported(0, None, None, "Kernel IR SSA identity overflow"))?;
            results.push(ValueDef::new(id, ty));
        }
        operations.push(Operation::new(results.clone(), kind));
        Ok(results)
    }

    fn require_local(
        &self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        local: u32,
    ) -> Result<usize, ProductionSemanticKirErrorV1> {
        let index = usize::try_from(local).map_err(|_| {
            unsupported(
                0,
                Some(block.index()),
                statement,
                "local does not fit this host",
            )
        })?;
        if index >= self.function.locals().len() {
            Err(unsupported(
                0,
                Some(block.index()),
                statement,
                "semantic local is out of range",
            ))
        } else {
            Ok(index)
        }
    }
}

fn strided_read_scalar_alignment_v1(element: &Type) -> Option<u32> {
    match element.as_scalar()? {
        ScalarType::Bool => Some(1),
        scalar => scalar
            .bit_width()
            .filter(|bits| bits % 8 == 0)
            .map(|bits| u32::from(bits / 8)),
    }
}

fn unsupported_terminator_detail(terminator: &SemanticTerminatorKindV1) -> &'static str {
    match terminator {
        SemanticTerminatorKindV1::SwitchInt { .. } => {
            "semantic switch-int terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::Call(_) => {
            "semantic call terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::TailCall(_) => {
            "semantic tail-call terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::Drop { .. } => {
            "semantic drop terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::Assert { .. } => {
            "semantic assert terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::FalseEdge { .. } => {
            "semantic false-edge terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::UnwindResume => {
            "semantic unwind-resume terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::UnwindTerminate => {
            "semantic unwind-terminate terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::Abort => {
            "semantic abort terminator has no exact Kernel IR lowering rule"
        }
        SemanticTerminatorKindV1::Goto(_)
        | SemanticTerminatorKindV1::Return
        | SemanticTerminatorKindV1::Unreachable => {
            "internally supported semantic terminator reached unsupported diagnostics"
        }
    }
}

fn unsupported_statement_detail(statement: &SemanticStatementKindV1) -> Option<&'static str> {
    match statement {
        SemanticStatementKindV1::Assign(assignment) => {
            Some(unsupported_rvalue_detail(assignment.value().kind()))
        }
        SemanticStatementKindV1::Store(_) => {
            Some("semantic store has no exact Kernel IR lowering rule")
        }
        SemanticStatementKindV1::AtomicRmw(_) => {
            Some("semantic atomic-rmw has no exact Kernel IR lowering rule")
        }
        SemanticStatementKindV1::AtomicCompareExchange(_) => {
            Some("semantic compare-exchange has no exact Kernel IR lowering rule")
        }
        SemanticStatementKindV1::SetDiscriminant { .. } => {
            Some("semantic set-discriminant has no exact Kernel IR lowering rule")
        }
        SemanticStatementKindV1::Deinitialize(_) => {
            Some("semantic deinitialize has no exact Kernel IR lowering rule")
        }
        SemanticStatementKindV1::StorageLive(_)
        | SemanticStatementKindV1::StorageDead(_)
        | SemanticStatementKindV1::Assume(_)
        | SemanticStatementKindV1::Nop => None,
    }
}

fn unsupported_rvalue_detail(value: &SemanticRvalueKindV1) -> &'static str {
    match value {
        SemanticRvalueKindV1::Use(_) => {
            "semantic assignment/use has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Unary { .. } => {
            "semantic assignment/unary has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Binary { .. } => {
            "semantic assignment/binary has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::CheckedBinary(_) => {
            "internally supported semantic checked arithmetic reached unsupported diagnostics"
        }
        SemanticRvalueKindV1::UncheckedBinary(_) => {
            "semantic unchecked arithmetic lacks an admitted overflow proof"
        }
        SemanticRvalueKindV1::Cast { .. } => {
            "semantic assignment/cast has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Borrow { .. } => {
            "semantic assignment/borrow has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::AddressOf { .. } => {
            "semantic assignment/address-of has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Length(_) => {
            "semantic assignment/length has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Discriminant(_) => {
            "semantic assignment/discriminant has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Aggregate(_) => {
            "semantic assignment/aggregate has no exact Kernel IR lowering rule"
        }
        SemanticRvalueKindV1::Load(_) => {
            "semantic assignment/load has no exact Kernel IR lowering rule"
        }
    }
}

fn lower_parameter_type(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<Type, ProductionSemanticKirErrorV1> {
    let shape = types
        .get(usize::try_from(ty.index()).unwrap_or(usize::MAX))
        .ok_or_else(|| unsupported(0, None, None, "kernel argument type is missing"))?
        .shape();
    if let Some((element, _)) = disjoint_slice_descriptor(callables, ty) {
        return Ok(Type::slice(
            lower_scalar_type(types, element)?,
            AddressSpace::Global,
            AccessMode::ReadWrite,
        ));
    }
    match shape {
        SemanticTypeShapeV1::Pointer(pointer) => {
            let access = match pointer.mutability() {
                SemanticMutabilityV1::Immutable => AccessMode::ReadOnly,
                SemanticMutabilityV1::Mutable => AccessMode::ReadWrite,
            };
            let address_space = lower_address_space(pointer.address_space())?;
            match pointer.metadata() {
                SemanticPointerMetadataV1::None => Ok(Type::pointer(
                    lower_memory_element_type(types, pointer.pointee())?,
                    address_space,
                    access,
                )),
                SemanticPointerMetadataV1::SliceLength => {
                    let pointee =
                        types
                            .get(pointer.pointee().index() as usize)
                            .ok_or_else(|| {
                                unsupported(0, None, None, "slice pointee type is missing")
                            })?;
                    let SemanticTypeShapeV1::Slice { element } = pointee.shape() else {
                        return Err(unsupported(
                            0,
                            None,
                            None,
                            "slice-length pointer metadata has a non-slice pointee",
                        ));
                    };
                    Ok(Type::slice(
                        lower_scalar_type(types, *element)?,
                        address_space,
                        access,
                    ))
                }
                SemanticPointerMetadataV1::VTable => Err(unsupported(
                    0,
                    None,
                    None,
                    "vtable-bearing kernel arguments are unsupported",
                )),
            }
        }
        SemanticTypeShapeV1::Scalar(_) => Ok(lower_scalar_type(types, ty)?),
        _ => Err(unsupported(
            0,
            None,
            None,
            "kernel argument type has no authenticated Kernel IR representation",
        )),
    }
}

fn lower_memory_element_type(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<Type, ProductionSemanticKirErrorV1> {
    if let Ok(scalar) = lower_scalar_type(types, ty) {
        return Ok(scalar);
    }
    transparent_scalar_storage_type(types, ty).ok_or_else(|| {
        ProductionSemanticKirErrorV1::ScalarTypeUnavailable {
            semantic_type: ty.index(),
            shape: types
                .get(ty.index() as usize)
                .map(|declaration| format!("{declaration:?}"))
                .unwrap_or_else(|| "<missing>".to_owned()),
        }
    })
}

fn lower_dynamic_lds_element_type_v1(
    types: &[SemanticTypeDeclV1],
    storage: SemanticTypeIdV1,
) -> Result<Type, ProductionSemanticKirErrorV1> {
    let mut current = storage;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) || visited.len() > MAX_SSA_VALUE_COMPONENTS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "exact LDS storage wrapper is recursive or too deep",
            ));
        }
        if let Ok(scalar) = lower_scalar_type(types, current) {
            return Ok(scalar);
        }
        let declaration = types
            .get(current.index() as usize)
            .ok_or_else(|| unsupported(0, None, None, "exact LDS storage type is missing"))?;
        let (fields, aggregate_layout) = match declaration.shape() {
            SemanticTypeShapeV1::Aggregate(aggregate) => {
                let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details()
                else {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "exact LDS aggregate storage lacks layout details",
                    ));
                };
                (aggregate.fields(), Some(layout))
            }
            SemanticTypeShapeV1::Union(union) => (union.fields(), None),
            _ => {
                return Err(unsupported(
                    0,
                    None,
                    None,
                    "exact LDS supports only scalar transparent storage",
                ));
            }
        };
        let mut candidate = None;
        for (index, field) in fields.iter().copied().enumerate() {
            let field_declaration = types
                .get(field.index() as usize)
                .ok_or_else(|| unsupported(0, None, None, "exact LDS storage field is missing"))?;
            if declaration.layout().size_bytes() == field_declaration.layout().size_bytes()
                && declaration.layout().alignment_bytes()
                    == field_declaration.layout().alignment_bytes()
                && !field_declaration.layout().is_uninhabited()
            {
                if candidate.replace((index, field)).is_some() {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "exact LDS storage has multiple layout-preserving fields",
                    ));
                }
            } else if field_declaration.layout().size_bytes() != Some(0) {
                return Err(unsupported(
                    0,
                    None,
                    None,
                    "exact LDS storage has a nontransparent alternate field",
                ));
            }
        }
        let (index, field) = candidate.ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "exact LDS storage has no layout-preserving field",
            )
        })?;
        if aggregate_layout.is_some_and(|layout| {
            layout.field_offsets().get(index) != Some(&0) || !layout.padding().is_empty()
        }) {
            return Err(unsupported(
                0,
                None,
                None,
                "exact LDS aggregate storage is not transparent",
            ));
        }
        let field_declaration = types
            .get(field.index() as usize)
            .ok_or_else(|| unsupported(0, None, None, "exact LDS storage field is missing"))?;
        if declaration.layout().size_bytes() != field_declaration.layout().size_bytes()
            || declaration.layout().alignment_bytes()
                != field_declaration.layout().alignment_bytes()
            || declaration.layout().is_uninhabited()
            || field_declaration.layout().is_uninhabited()
        {
            return Err(unsupported(
                0,
                None,
                None,
                "exact LDS storage wrapper changes scalar layout",
            ));
        }
        current = field;
    }
}

fn transparent_scalar_storage_type(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Option<Type> {
    let mut current = ty;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) || visited.len() > MAX_SSA_VALUE_COMPONENTS_V1 {
            return None;
        }
        if let Ok(scalar) = lower_scalar_type(types, current) {
            return Some(scalar);
        }
        let declaration = types.get(current.index() as usize)?;
        let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
            return None;
        };
        let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
            return None;
        };
        if declaration.layout().is_uninhabited()
            || aggregate.fields().len() != 1
            || layout.field_offsets() != [0]
            || !layout.padding().is_empty()
        {
            return None;
        }
        let field_ty = aggregate.fields()[0];
        let field = types.get(field_ty.index() as usize)?;
        if field.layout().is_uninhabited()
            || declaration.layout().size_bytes() != field.layout().size_bytes()
            || declaration.layout().alignment_bytes() != field.layout().alignment_bytes()
        {
            return None;
        }
        current = field_ty;
    }
}

fn lower_kernel_parameter_type(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    function: &SemanticFunctionDeclV1,
    argument: u32,
    ty: SemanticTypeIdV1,
) -> Result<Type, ProductionSemanticKirErrorV1> {
    let declaration = types
        .get(usize::try_from(ty.index()).unwrap_or(usize::MAX))
        .ok_or_else(|| unsupported(0, None, None, "kernel argument type is missing"))?;
    if let Some(parameter) =
        authenticated_disjoint_slice_parameter(types, callables, function, argument, ty)
    {
        return Ok(parameter);
    }
    if !matches!(declaration.shape(), SemanticTypeShapeV1::Aggregate(_)) {
        return lower_parameter_type(types, callables, ty);
    }
    authenticated_global_mut_pointer_parameter(types, function, argument, ty).ok_or_else(|| {
        unsupported(
            0,
            None,
            None,
            "kernel argument type has no authenticated Kernel IR representation",
        )
    })
}

/// Rechecks the compiler-issued `DisjointSlice<T, IndexSpace>` source, ABI,
/// ownership, and layout facts before assigning writable global-slice meaning
/// at the Kernel IR boundary.
fn authenticated_disjoint_slice_parameter(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    function: &SemanticFunctionDeclV1,
    argument: u32,
    ty: SemanticTypeIdV1,
) -> Option<Type> {
    let (element, raw_index) = disjoint_slice_descriptor(callables, ty)?;
    let argument = usize::try_from(argument).ok()?;
    let abi = function.abi();
    if abi.source_input_types().get(argument) != Some(&ty)
        || abi.source_argument_ownership().get(argument)
            != Some(&SemanticSourceArgumentOwnershipV1::ExclusiveOwner)
    {
        return None;
    }
    let abi_argument = abi.adjusted_arguments().get(argument)?;
    if abi_argument.ty() != ty
        || abi_argument.value().adjusted().is_some()
        || !matches!(abi_argument.mode(), SemanticAbiPassModeV1::Pair { .. })
    {
        return None;
    }

    let declaration = types.get(ty.index() as usize)?;
    let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
        return None;
    };
    let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
        return None;
    };
    let SemanticBackendReprV1::ScalarPair { first, second } = declaration.layout().backend_repr()
    else {
        return None;
    };
    if aggregate.fields().len() != layout.field_offsets().len() {
        return None;
    }

    let mut pointer_field = None;
    let mut length_field = None;
    for (index, (&field_ty, &offset)) in aggregate
        .fields()
        .iter()
        .zip(layout.field_offsets())
        .enumerate()
    {
        let field = types.get(field_ty.index() as usize)?;
        if let SemanticTypeShapeV1::Pointer(pointer) = field.shape()
            && pointer.pointee() == element
            && pointer.kind() == SemanticPointerKindV1::Raw
            && pointer.mutability() == SemanticMutabilityV1::Mutable
            && pointer.address_space() == 0
            && pointer.pointer_width_bits() == 64
            && pointer.metadata() == SemanticPointerMetadataV1::None
        {
            let SemanticBackendReprV1::Scalar(pointer_scalar) = field.layout().backend_repr()
            else {
                return None;
            };
            if pointer_field.replace(index).is_some() || offset != 0 || pointer_scalar != first {
                return None;
            }
        } else if field_ty == raw_index {
            let SemanticBackendReprV1::Scalar(length_scalar) = field.layout().backend_repr() else {
                return None;
            };
            if length_field.replace(index).is_some()
                || offset != 8
                || length_scalar != second
                || !matches!(
                    field.shape(),
                    SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                        signed: false,
                        bits: 64
                    })
                )
            {
                return None;
            }
        } else if field.layout().size_bytes() != Some(0) || field.layout().is_uninhabited() {
            return None;
        }
    }
    pointer_field?;
    length_field?;
    if declaration.layout().size_bytes() != Some(16) || declaration.layout().alignment_bytes() != 8
    {
        return None;
    }

    Some(Type::slice(
        lower_scalar_type(types, element).ok()?,
        AddressSpace::Global,
        AccessMode::ReadWrite,
    ))
}

/// Recognizes the exact source/ABI/layout contract established for the
/// production `DeviceGlobalMutPtr<T>` argument. No individual structural fact
/// is authority: the production descriptor join has already authenticated the
/// source wrapper, and this boundary rechecks every retained consequence before
/// assigning the Kernel IR global address space.
fn authenticated_global_mut_pointer_parameter(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    argument: u32,
    ty: SemanticTypeIdV1,
) -> Option<Type> {
    let argument = usize::try_from(argument).ok()?;
    let abi = function.abi();
    if abi.source_input_types().get(argument) != Some(&ty)
        || abi.source_argument_ownership().get(argument)
            != Some(&SemanticSourceArgumentOwnershipV1::ExclusiveOwner)
    {
        return None;
    }
    let abi_argument = abi.adjusted_arguments().get(argument)?;
    if abi_argument.ty() != ty || !matches!(abi_argument.mode(), SemanticAbiPassModeV1::Direct(_)) {
        return None;
    }

    let declaration = types.get(ty.index() as usize)?;
    let pointee = abi_argument
        .value()
        .pointee_override()
        .or(declaration.abi_properties().first_pointee())?;
    if pointee.kind() != SemanticAbiPointeeKindV1::Raw {
        return None;
    }
    let outer_pointer = scalar_backend_pointer(declaration)?;
    let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
        return None;
    };
    let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
        return None;
    };
    if aggregate.fields().len() != layout.field_offsets().len() {
        return None;
    }

    let mut physical_pointer = None;
    for (&field_ty, &field_offset) in aggregate.fields().iter().zip(layout.field_offsets()) {
        let field = types.get(field_ty.index() as usize)?;
        if field.layout().size_bytes() == Some(0) {
            if field.layout().is_uninhabited() {
                return None;
            }
            continue;
        }
        if physical_pointer.is_some() || field_offset != 0 {
            return None;
        }
        let SemanticTypeShapeV1::Pointer(pointer) = field.shape() else {
            return None;
        };
        if pointer.kind() != SemanticPointerKindV1::Raw
            || pointer.mutability() != SemanticMutabilityV1::Mutable
            || pointer.metadata() != SemanticPointerMetadataV1::None
            || scalar_backend_pointer(field)? != outer_pointer
            || declaration.layout().size_bytes() != field.layout().size_bytes()
            || declaration.layout().alignment_bytes() != field.layout().alignment_bytes()
        {
            return None;
        }
        physical_pointer = Some(pointer);
    }

    let pointer = physical_pointer?;
    Some(Type::pointer(
        lower_scalar_type(types, pointer.pointee()).ok()?,
        AddressSpace::Global,
        AccessMode::ReadWrite,
    ))
}

fn scalar_backend_pointer(declaration: &SemanticTypeDeclV1) -> Option<SemanticBackendPrimitiveV1> {
    let SemanticBackendReprV1::Scalar(scalar) = declaration.layout().backend_repr() else {
        return None;
    };
    let primitive = scalar.primitive();
    matches!(primitive, SemanticBackendPrimitiveV1::Pointer { .. }).then_some(primitive)
}

const MAX_SSA_VALUE_COMPONENTS_V1: usize = 256;
const MAX_ENUM_PAYLOAD_STORAGE_COMPONENTS_V1: usize = 4_096;

fn lower_ssa_value_components_v1(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<Vec<(SemanticTypeIdV1, Type)>, ProductionSemanticKirErrorV1> {
    fn append(
        types: &[SemanticTypeDeclV1],
        ty: SemanticTypeIdV1,
        output: &mut Vec<(SemanticTypeIdV1, Type)>,
        structural_nodes: &mut usize,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        *structural_nodes = structural_nodes.checked_add(1).ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "aggregate SSA value exceeds the structural limit",
            )
        })?;
        if *structural_nodes > MAX_SSA_VALUE_COMPONENTS_V1
            || output.len() > MAX_SSA_VALUE_COMPONENTS_V1
        {
            return Err(unsupported(
                0,
                None,
                None,
                "aggregate SSA value exceeds the structural or component limit",
            ));
        }
        let shape = types
            .get(ty.index() as usize)
            .ok_or_else(|| unsupported(0, None, None, "aggregate SSA type is missing"))?
            .shape();
        match shape {
            SemanticTypeShapeV1::Unit => Ok(()),
            SemanticTypeShapeV1::Enum { discriminant, .. } => {
                output.push((*discriminant, lower_scalar_type(types, *discriminant)?));
                Ok(())
            }
            SemanticTypeShapeV1::Scalar(_) | SemanticTypeShapeV1::ValidityScalar(_) => {
                output.push((ty, lower_scalar_type(types, ty)?));
                Ok(())
            }
            SemanticTypeShapeV1::Pointer(_) => {
                output.push((ty, lower_parameter_type(types, &[], ty)?));
                Ok(())
            }
            SemanticTypeShapeV1::Array { element, length } => {
                let length = usize::try_from(*length).map_err(|_| {
                    unsupported(0, None, None, "aggregate SSA array length is too large")
                })?;
                if length > MAX_SSA_VALUE_COMPONENTS_V1 {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "aggregate SSA array length is too large",
                    ));
                }
                for _ in 0..length {
                    append(types, *element, output, structural_nodes)?;
                }
                Ok(())
            }
            SemanticTypeShapeV1::Tuple(fields) | SemanticTypeShapeV1::Aggregate(fields) => {
                for field in fields.fields() {
                    append(types, *field, output, structural_nodes)?;
                }
                Ok(())
            }
            _ => Err(unsupported(
                0,
                None,
                None,
                "type has no bounded aggregate SSA representation",
            )),
        }
    }

    let mut output = Vec::new();
    let mut structural_nodes = 0;
    append(types, ty, &mut output, &mut structural_nodes)?;
    if output.len() > MAX_SSA_VALUE_COMPONENTS_V1 {
        return Err(unsupported(
            0,
            None,
            None,
            "aggregate SSA value exceeds the component limit",
        ));
    }
    Ok(output)
}

fn lower_ssa_value_types(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
    Ok(lower_ssa_value_components_v1(types, ty)?
        .into_iter()
        .map(|(_, ty)| ty)
        .collect())
}

fn lower_workgroup_collective_scratch_transport_v1(
    types: &[SemanticTypeDeclV1],
    scratch: SemanticTypeIdV1,
    element: SemanticTypeIdV1,
) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
    let scalar = lower_scalar_type(types, element)?;
    if !matches!(
        scalar,
        Type::Scalar(ScalarType::U32 | ScalarType::I32 | ScalarType::F32)
    ) {
        return Err(unsupported(
            0,
            None,
            None,
            "promoted workgroup scratch element is unsupported",
        ));
    }
    let components = lower_ssa_value_components_v1(types, scratch)?;
    let [(pointer_semantic_type, Type::Pointer(pointer)), (_, slots)] = components.as_slice()
    else {
        return Err(unsupported(
            0,
            None,
            None,
            "promoted workgroup scratch transport shape changed",
        ));
    };
    let pointer_declaration = types
        .get(pointer_semantic_type.index() as usize)
        .ok_or_else(|| unsupported(0, None, None, "workgroup scratch pointer type is missing"))?;
    let SemanticTypeShapeV1::Pointer(semantic_pointer) = pointer_declaration.shape() else {
        return Err(unsupported(
            0,
            None,
            None,
            "workgroup scratch pointer field is not a semantic pointer",
        ));
    };
    if semantic_pointer.pointee() != element
        || semantic_pointer.kind() != SemanticPointerKindV1::Raw
        || semantic_pointer.mutability() != SemanticMutabilityV1::Mutable
        || semantic_pointer.address_space() != 0
        || semantic_pointer.pointer_width_bits() != 64
        || semantic_pointer.metadata() != SemanticPointerMetadataV1::None
        || pointer.address_space != AddressSpace::Global
        || pointer.access != AccessMode::ReadWrite
        || *pointer.pointee != scalar
        || slots != &Type::Scalar(ScalarType::U32)
    {
        return Err(unsupported(
            0,
            None,
            None,
            "promoted workgroup scratch source ABI changed",
        ));
    }
    Ok(vec![
        Type::pointer(scalar, AddressSpace::Workgroup, AccessMode::ReadWrite),
        slots.clone(),
    ])
}

fn plan_enum_payload_storage_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    control_flow_ssa: &SemanticControlFlowSsaPlanV1,
    sources: &BTreeMap<(u32, u32, u32), SemanticEnumPayloadSourceV1>,
    next_value: &mut u32,
) -> Result<
    (
        BTreeMap<(u32, u32, u32), SemanticEnumPayloadFieldStorageV1>,
        BTreeSet<(u32, u32, u32)>,
    ),
    ProductionSemanticKirErrorV1,
> {
    let mut storage = BTreeMap::new();
    let mut requires_compile_time_custody = BTreeSet::new();
    let mut component_count = 0_usize;
    for (local, promoted) in &control_flow_ssa.promoted {
        let declaration = types
            .get(promoted.semantic_type.index() as usize)
            .ok_or_else(|| unsupported(0, None, None, "promoted enum type is missing"))?;
        let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
            continue;
        };
        if function.locals().get(*local as usize).is_none() {
            return Err(unsupported(0, None, None, "promoted enum local is missing"));
        }
        for (variant, definition) in variants.iter().enumerate() {
            for (field, semantic_type) in definition.fields().fields().iter().copied().enumerate() {
                let mut components = Vec::new();
                let key = (*local, variant as u32, field as u32);
                let exact_enum_variant = sources.get(&key).and_then(|source| {
                    exact_enum_variant_for_source_v1(function, source, semantic_type)
                });
                let compiler_issued_binding = exact_enum_variant
                    .is_none()
                    .then(|| {
                        control_flow_ssa
                            .compiler_issued_bindings
                            .get(&semantic_type)
                            .copied()
                    })
                    .flatten();
                let component_types = match exact_enum_variant {
                    Some(exact_variant) => {
                        lower_exact_enum_components_v1(types, semantic_type, exact_variant)
                    }
                    None => match compiler_issued_binding {
                        Some(
                            binding @ SemanticPromotedBindingV1::WorkgroupCollectiveScratch {
                                ..
                            },
                        ) => {
                            let semantic_components =
                                lower_ssa_value_components_v1(types, semantic_type)?;
                            let transport = binding.transport_types(types, semantic_type)?;
                            if semantic_components.len() != transport.len() {
                                return Err(unsupported(
                                    0,
                                    None,
                                    None,
                                    "compiler-issued enum payload transport arity changed",
                                ));
                            }
                            Ok(semantic_components
                                .into_iter()
                                .zip(transport)
                                .map(|((semantic_type, _), transport)| (semantic_type, transport))
                                .collect())
                        }
                        Some(
                            SemanticPromotedBindingV1::Ordinary
                            | SemanticPromotedBindingV1::MatrixFragment { .. }
                            | SemanticPromotedBindingV1::AccumulatorFragment { .. }
                            | SemanticPromotedBindingV1::Gfx950LdsTransposeTile { .. },
                        )
                        | None => lower_ssa_value_components_v1(types, semantic_type),
                    },
                };
                let component_types = match component_types {
                    Ok(components) => components,
                    Err(ProductionSemanticKirErrorV1::Unsupported {
                        detail: "type has no bounded aggregate SSA representation",
                        ..
                    }) => continue,
                    Err(error) => return Err(error),
                };
                if component_types
                    .iter()
                    .any(|(_, kernel_type)| !kernel_type.is_storable())
                {
                    if sources.contains_key(&key) {
                        requires_compile_time_custody.insert(key);
                        continue;
                    }
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "enum payload component is not storable in private memory and has no unique source",
                    ));
                }
                for (component_type, kernel_type) in component_types {
                    component_count = component_count.checked_add(1).ok_or_else(|| {
                        unsupported(0, None, None, "enum payload storage count overflow")
                    })?;
                    if component_count > MAX_ENUM_PAYLOAD_STORAGE_COMPONENTS_V1 {
                        return Err(unsupported(
                            0,
                            None,
                            None,
                            "enum payload storage exceeds the component limit",
                        ));
                    }
                    let alignment = types
                        .get(component_type.index() as usize)
                        .and_then(|ty| u32::try_from(ty.layout().alignment_bytes()).ok())
                        .filter(|alignment| *alignment != 0)
                        .ok_or_else(|| {
                            unsupported(
                                0,
                                None,
                                None,
                                "enum payload component alignment is unsupported",
                            )
                        })?;
                    let pointer = ValueId(*next_value);
                    *next_value = next_value.checked_add(1).ok_or_else(|| {
                        unsupported(0, None, None, "enum payload SSA identity overflow")
                    })?;
                    components.push(SemanticEnumPayloadComponentStorageV1 {
                        pointer,
                        kernel_type,
                        alignment,
                    });
                }
                storage.insert(
                    key,
                    SemanticEnumPayloadFieldStorageV1 {
                        semantic_type,
                        exact_enum_variant,
                        compiler_issued_binding,
                        components: components.into_boxed_slice(),
                    },
                );
            }
        }
    }
    Ok((storage, requires_compile_time_custody))
}

fn plan_unique_enum_payload_sources_v1(
    function: &SemanticFunctionDeclV1,
    control_flow_ssa: &SemanticControlFlowSsaPlanV1,
) -> BTreeMap<(u32, u32, u32), SemanticEnumPayloadSourceV1> {
    let mut candidates = BTreeMap::<(u32, u32, u32), Option<SemanticEnumPayloadSourceV1>>::new();
    for block in function.blocks() {
        for statement in block.statements() {
            let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                continue;
            };
            let destination = assignment.destination();
            if !destination.projections().is_empty()
                || !control_flow_ssa
                    .promoted
                    .contains_key(&destination.local().index())
            {
                continue;
            }
            let SemanticRvalueKindV1::Aggregate(aggregate) = assignment.value().kind() else {
                continue;
            };
            let SemanticAggregateKindV1::EnumVariant(variant) = aggregate.kind() else {
                continue;
            };
            for (field, operand) in aggregate.operands().iter().enumerate() {
                let key = (destination.local().index(), *variant, field as u32);
                let source = match operand {
                    SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => {
                        Some(SemanticEnumPayloadSourceV1 {
                            place: place.clone(),
                        })
                    }
                    SemanticOperandV1::Constant(_) => None,
                };
                match candidates.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(source);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry.insert(None);
                    }
                }
            }
        }
    }
    candidates
        .into_iter()
        .filter_map(|(key, source)| source.map(|source| (key, source)))
        .collect()
}

fn exact_enum_variant_for_source_v1(
    function: &SemanticFunctionDeclV1,
    source: &SemanticEnumPayloadSourceV1,
    semantic_type: SemanticTypeIdV1,
) -> Option<u32> {
    if !source.place.projections().is_empty() || source.place.ty() != semantic_type {
        return None;
    }
    let local = source.place.local();
    let mut definitions = 0_u32;
    let mut variant = None;
    for block in function.blocks() {
        for statement in block.statements() {
            let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                continue;
            };
            if !assignment.destination().projections().is_empty()
                || assignment.destination().local() != local
            {
                continue;
            }
            definitions = definitions.saturating_add(1);
            variant = match assignment.value().kind() {
                SemanticRvalueKindV1::Aggregate(aggregate)
                    if assignment.value().result_type() == semantic_type =>
                {
                    match aggregate.kind() {
                        SemanticAggregateKindV1::EnumVariant(variant) => Some(*variant),
                        SemanticAggregateKindV1::Array
                        | SemanticAggregateKindV1::Tuple
                        | SemanticAggregateKindV1::Aggregate => None,
                    }
                }
                _ => None,
            };
        }
        if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
            && call.destination().is_some_and(|destination| {
                destination.place().projections().is_empty() && destination.place().local() == local
            })
        {
            definitions = definitions.saturating_add(1);
            variant = None;
        }
    }
    (definitions == 1).then_some(variant).flatten()
}

fn lower_exact_enum_components_v1(
    types: &[SemanticTypeDeclV1],
    semantic_type: SemanticTypeIdV1,
    variant: u32,
) -> Result<Vec<(SemanticTypeIdV1, Type)>, ProductionSemanticKirErrorV1> {
    let (discriminant, variants) = semantic_enum_shape(types, semantic_type)?;
    let selected = variants.get(variant as usize).ok_or_else(|| {
        unsupported(
            0,
            None,
            None,
            "exact enum payload source variant is out of range",
        )
    })?;
    let mut components = vec![(discriminant, lower_scalar_type(types, discriminant)?)];
    for field in selected.fields().fields() {
        components.extend(lower_ssa_value_components_v1(types, *field)?);
        if components.len() > MAX_SSA_VALUE_COMPONENTS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "exact enum payload source exceeds the component limit",
            ));
        }
    }
    Ok(components)
}

fn exact_enum_binding_values_v1(
    binding: &SemanticValueBindingV1,
    expected_variant: u32,
) -> Result<Vec<(ValueId, Type)>, &'static str> {
    let SemanticValueBindingV1::Enum {
        discriminant,
        discriminant_ty,
        variant: Some(actual_variant),
        payloads,
        ..
    } = binding
    else {
        return Err("exact enum payload source is not a variant-refined enum");
    };
    if *actual_variant != expected_variant {
        return Err("exact enum payload source variant changed");
    }
    let fields = payloads
        .get(actual_variant)
        .ok_or("exact enum payload source fields are unavailable")?;
    let mut values = vec![(*discriminant, discriminant_ty.clone())];
    for field in fields {
        field.append_values(&mut values)?;
    }
    Ok(values)
}

fn binding_from_exact_enum_value_defs_v1(
    types: &[SemanticTypeDeclV1],
    semantic_type: SemanticTypeIdV1,
    variant: u32,
    values: &[ValueDef],
) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
    let (discriminant_type, variants) = semantic_enum_shape(types, semantic_type)?;
    let definition = variants
        .get(variant as usize)
        .ok_or_else(|| unsupported(0, None, None, "exact enum storage variant is out of range"))?;
    let discriminant = values
        .first()
        .ok_or_else(|| unsupported(0, None, None, "exact enum storage is truncated"))?;
    let expected_discriminant = lower_scalar_type(types, discriminant_type)?;
    if discriminant.ty != expected_discriminant {
        return Err(unsupported(
            0,
            None,
            None,
            "exact enum storage discriminant type changed",
        ));
    }
    let mut cursor = 1_usize;
    let mut fields = Vec::with_capacity(definition.fields().fields().len());
    for field_type in definition.fields().fields() {
        let component_count = lower_ssa_value_components_v1(types, *field_type)?.len();
        let end = cursor.checked_add(component_count).ok_or_else(|| {
            unsupported(0, None, None, "exact enum storage component count overflow")
        })?;
        let components = values
            .get(cursor..end)
            .ok_or_else(|| unsupported(0, None, None, "exact enum storage is truncated"))?;
        fields.push(binding_from_value_defs(types, *field_type, components)?);
        cursor = end;
    }
    if cursor != values.len() {
        return Err(unsupported(
            0,
            None,
            None,
            "exact enum storage has trailing components",
        ));
    }
    Ok(SemanticValueBindingV1::Enum {
        discriminant: discriminant.id,
        discriminant_ty: discriminant.ty.clone(),
        semantic_type,
        variant: Some(variant),
        payloads: BTreeMap::from([(variant, fields)]),
    })
}

fn binding_from_value_defs(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    values: &[ValueDef],
) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
    binding_from_value_defs_with_validation(types, ty, values, true)
}

fn binding_from_value_defs_with_validation(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    values: &[ValueDef],
    validate_scalar_types: bool,
) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
    fn build(
        types: &[SemanticTypeDeclV1],
        ty: SemanticTypeIdV1,
        values: &[ValueDef],
        cursor: &mut usize,
        structural_nodes: &mut usize,
        validate_scalar_types: bool,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        *structural_nodes = structural_nodes.checked_add(1).ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "aggregate SSA binding exceeds the structural limit",
            )
        })?;
        if *structural_nodes > MAX_SSA_VALUE_COMPONENTS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "aggregate SSA binding exceeds the structural limit",
            ));
        }
        let shape = types
            .get(ty.index() as usize)
            .ok_or_else(|| unsupported(0, None, None, "aggregate SSA type is missing"))?
            .shape();
        match shape {
            SemanticTypeShapeV1::Unit => Ok(SemanticValueBindingV1::Unit),
            SemanticTypeShapeV1::Enum { discriminant, .. } => {
                let value = values.get(*cursor).ok_or_else(|| {
                    unsupported(0, None, None, "enum SSA discriminant is truncated")
                })?;
                let expected = lower_scalar_type(types, *discriminant)?;
                if validate_scalar_types && value.ty != expected {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "enum SSA discriminant type changed",
                    ));
                }
                *cursor += 1;
                Ok(SemanticValueBindingV1::Enum {
                    discriminant: value.id,
                    discriminant_ty: value.ty.clone(),
                    semantic_type: ty,
                    variant: None,
                    payloads: BTreeMap::new(),
                })
            }
            SemanticTypeShapeV1::Scalar(_) | SemanticTypeShapeV1::ValidityScalar(_) => {
                let value = values.get(*cursor).ok_or_else(|| {
                    unsupported(0, None, None, "aggregate SSA value is truncated")
                })?;
                let expected = lower_scalar_type(types, ty)?;
                if validate_scalar_types && value.ty != expected {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "aggregate SSA component type changed",
                    ));
                }
                *cursor += 1;
                Ok(SemanticValueBindingV1::Value {
                    id: value.id,
                    ty: value.ty.clone(),
                })
            }
            SemanticTypeShapeV1::Pointer(_) => {
                let value = values.get(*cursor).ok_or_else(|| {
                    unsupported(0, None, None, "aggregate SSA pointer is truncated")
                })?;
                if validate_scalar_types {
                    let expected = lower_parameter_type(types, &[], ty)?;
                    if value.ty != expected {
                        return Err(unsupported(
                            0,
                            None,
                            None,
                            "aggregate SSA pointer component type changed",
                        ));
                    }
                }
                *cursor += 1;
                Ok(SemanticValueBindingV1::Value {
                    id: value.id,
                    ty: value.ty.clone(),
                })
            }
            SemanticTypeShapeV1::Array { element, length } => {
                let length = usize::try_from(*length).map_err(|_| {
                    unsupported(0, None, None, "aggregate SSA array length is too large")
                })?;
                if length > MAX_SSA_VALUE_COMPONENTS_V1 {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "aggregate SSA array length is too large",
                    ));
                }
                let mut fields = Vec::with_capacity(length);
                for _ in 0..length {
                    fields.push(build(
                        types,
                        *element,
                        values,
                        cursor,
                        structural_nodes,
                        validate_scalar_types,
                    )?);
                }
                Ok(SemanticValueBindingV1::Aggregate(fields))
            }
            SemanticTypeShapeV1::Tuple(fields) | SemanticTypeShapeV1::Aggregate(fields) => {
                let mut bindings = Vec::with_capacity(fields.fields().len());
                for field in fields.fields() {
                    bindings.push(build(
                        types,
                        *field,
                        values,
                        cursor,
                        structural_nodes,
                        validate_scalar_types,
                    )?);
                }
                Ok(SemanticValueBindingV1::Aggregate(bindings))
            }
            _ => Err(unsupported(
                0,
                None,
                None,
                "type has no bounded aggregate SSA representation",
            )),
        }
    }

    let mut cursor = 0;
    let mut structural_nodes = 0;
    let binding = build(
        types,
        ty,
        values,
        &mut cursor,
        &mut structural_nodes,
        validate_scalar_types,
    )?;
    if cursor != values.len() {
        return Err(unsupported(
            0,
            None,
            None,
            "aggregate SSA value has trailing components",
        ));
    }
    Ok(binding)
}

fn lower_scalar_type(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<Type, ProductionSemanticKirErrorV1> {
    let shape = types
        .get(usize::try_from(ty.index()).unwrap_or(usize::MAX))
        .ok_or_else(|| unsupported(0, None, None, "scalar type is missing"))?
        .shape();
    let scalar = match shape {
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool) => ScalarType::Bool,
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer { signed, bits }) => {
            match (*signed, *bits) {
                (true, 8) => ScalarType::I8,
                (true, 16) => ScalarType::I16,
                (true, 32) => ScalarType::I32,
                (true, 64) => ScalarType::I64,
                (true, 128) => ScalarType::I128,
                (false, 8) => ScalarType::U8,
                (false, 16) => ScalarType::U16,
                (false, 32) => ScalarType::U32,
                (false, 64) => ScalarType::U64,
                (false, 128) => ScalarType::U128,
                _ => {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "integer argument width is unsupported",
                    ));
                }
            }
        }
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits }) => match bits {
            16 => ScalarType::F16,
            32 => ScalarType::F32,
            64 => ScalarType::F64,
            _ => {
                return Err(unsupported(
                    0,
                    None,
                    None,
                    "floating argument width is unsupported",
                ));
            }
        },
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Char) => ScalarType::U32,
        SemanticTypeShapeV1::ValidityScalar(validity) => {
            return lower_scalar_kind(validity.scalar());
        }
        _ => {
            return Err(ProductionSemanticKirErrorV1::ScalarTypeUnavailable {
                semantic_type: ty.index(),
                shape: types
                    .get(ty.index() as usize)
                    .map(|declaration| format!("{declaration:?}"))
                    .unwrap_or_else(|| "<missing>".to_owned()),
            });
        }
    };
    Ok(Type::Scalar(scalar))
}

const fn semantic_operand_type(operand: &SemanticOperandV1) -> SemanticTypeIdV1 {
    match operand {
        SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => place.ty(),
        SemanticOperandV1::Constant(constant) => constant.ty(),
    }
}

const fn lower_atomic_rmw_kind(
    operation: SemanticAtomicRmwOpV1,
    scalar: ScalarType,
) -> Option<AtomicKind> {
    if !scalar.is_integer() {
        return None;
    }
    Some(match operation {
        SemanticAtomicRmwOpV1::Exchange => AtomicKind::Exchange,
        SemanticAtomicRmwOpV1::Add => AtomicKind::Add,
        SemanticAtomicRmwOpV1::Subtract => AtomicKind::Subtract,
        SemanticAtomicRmwOpV1::BitAnd => AtomicKind::BitAnd,
        SemanticAtomicRmwOpV1::BitNand => return None,
        SemanticAtomicRmwOpV1::BitOr => AtomicKind::BitOr,
        SemanticAtomicRmwOpV1::BitXor => AtomicKind::BitXor,
        SemanticAtomicRmwOpV1::SignedMaximum if scalar.is_signed_integer() => AtomicKind::Max,
        SemanticAtomicRmwOpV1::SignedMinimum if scalar.is_signed_integer() => AtomicKind::Min,
        SemanticAtomicRmwOpV1::UnsignedMaximum if !scalar.is_signed_integer() => AtomicKind::Max,
        SemanticAtomicRmwOpV1::UnsignedMinimum if !scalar.is_signed_integer() => AtomicKind::Min,
        SemanticAtomicRmwOpV1::SignedMaximum
        | SemanticAtomicRmwOpV1::SignedMinimum
        | SemanticAtomicRmwOpV1::UnsignedMaximum
        | SemanticAtomicRmwOpV1::UnsignedMinimum => return None,
    })
}

const fn lower_atomic_ordering(ordering: SemanticAtomicOrderingV1) -> MemoryOrdering {
    match ordering {
        SemanticAtomicOrderingV1::Relaxed => MemoryOrdering::Relaxed,
        SemanticAtomicOrderingV1::Release => MemoryOrdering::Release,
        SemanticAtomicOrderingV1::Acquire => MemoryOrdering::Acquire,
        SemanticAtomicOrderingV1::AcquireRelease => MemoryOrdering::AcquireRelease,
        SemanticAtomicOrderingV1::SequentiallyConsistent => MemoryOrdering::SequentiallyConsistent,
    }
}

const fn lower_atomic_scope(scope: SemanticAtomicScopeV1) -> Option<SynchronizationScope> {
    match scope {
        SemanticAtomicScopeV1::Workgroup => Some(SynchronizationScope::Workgroup),
        SemanticAtomicScopeV1::Agent => Some(SynchronizationScope::Device),
        SemanticAtomicScopeV1::System => Some(SynchronizationScope::System),
        SemanticAtomicScopeV1::SingleThread | SemanticAtomicScopeV1::Device => None,
    }
}

fn checked_binary_result_type(
    types: &[SemanticTypeDeclV1],
    operand_type: SemanticTypeIdV1,
    result_type: SemanticTypeIdV1,
) -> Result<Type, &'static str> {
    let operand_shape = types
        .get(operand_type.index() as usize)
        .ok_or("semantic checked arithmetic operand type is missing")?
        .shape();
    let SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer { signed, bits }) = operand_shape
    else {
        return Err("semantic checked arithmetic operand is not a plain integer");
    };
    if !matches!(bits, 8 | 16 | 32 | 64 | 128) {
        return Err("semantic checked arithmetic integer width is unsupported");
    }
    let result_shape = types
        .get(result_type.index() as usize)
        .ok_or("semantic checked arithmetic result type is missing")?
        .shape();
    let SemanticTypeShapeV1::Tuple(fields) = result_shape else {
        return Err("semantic checked arithmetic result is not a tuple");
    };
    let [value_type, overflow_type] = fields.fields() else {
        return Err("semantic checked arithmetic result is not a two-field tuple");
    };
    if *value_type != operand_type {
        return Err("semantic checked arithmetic value result type differs from its operands");
    }
    if !matches!(
        types
            .get(overflow_type.index() as usize)
            .map(SemanticTypeDeclV1::shape),
        Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool))
    ) {
        return Err("semantic checked arithmetic overflow result is not bool");
    }
    lower_scalar_kind(SemanticScalarTypeV1::Integer {
        signed: *signed,
        bits: *bits,
    })
    .map_err(|_| "semantic checked arithmetic integer width is unsupported")
}

const fn lower_checked_binary(operation: SemanticCheckedBinaryOpV1) -> CheckedBinaryOperator {
    match operation {
        SemanticCheckedBinaryOpV1::Add => CheckedBinaryOperator::Add,
        SemanticCheckedBinaryOpV1::Subtract => CheckedBinaryOperator::Subtract,
        SemanticCheckedBinaryOpV1::Multiply => CheckedBinaryOperator::Multiply,
    }
}

const fn lower_f32_math_function(function: SemanticF32MathFunctionV1) -> F32MathFunction {
    match function {
        SemanticF32MathFunctionV1::Sqrt => F32MathFunction::Sqrt,
        SemanticF32MathFunctionV1::FusedMultiplyAdd => F32MathFunction::FusedMultiplyAdd,
        SemanticF32MathFunctionV1::Floor => F32MathFunction::Floor,
        SemanticF32MathFunctionV1::Ceil => F32MathFunction::Ceil,
        SemanticF32MathFunctionV1::Truncate => F32MathFunction::Truncate,
        SemanticF32MathFunctionV1::RoundTiesEven => F32MathFunction::RoundTiesEven,
        SemanticF32MathFunctionV1::Sin => F32MathFunction::Sin,
        SemanticF32MathFunctionV1::Cos => F32MathFunction::Cos,
        SemanticF32MathFunctionV1::Exp => F32MathFunction::Exp,
        SemanticF32MathFunctionV1::Exp2 => F32MathFunction::Exp2,
        SemanticF32MathFunctionV1::Ln => F32MathFunction::Ln,
        SemanticF32MathFunctionV1::Log2 => F32MathFunction::Log2,
        SemanticF32MathFunctionV1::Log10 => F32MathFunction::Log10,
    }
}

const fn lower_gfx950_lds_transpose_format_v1(
    format: SemanticGfx950LdsTransposeFormatV1,
) -> Gfx950LdsTransposeFormatV1 {
    match format {
        SemanticGfx950LdsTransposeFormatV1::Fp4E2M1 => Gfx950LdsTransposeFormatV1::Fp4E2M1,
        SemanticGfx950LdsTransposeFormatV1::Fp8E4M3 => Gfx950LdsTransposeFormatV1::Fp8E4M3,
    }
}

const fn semantic_gfx950_lds_transpose_format_v1(
    format: Gfx950LdsTransposeFormatV1,
) -> SemanticGfx950LdsTransposeFormatV1 {
    match format {
        Gfx950LdsTransposeFormatV1::Fp4E2M1 => SemanticGfx950LdsTransposeFormatV1::Fp4E2M1,
        Gfx950LdsTransposeFormatV1::Fp8E4M3 => SemanticGfx950LdsTransposeFormatV1::Fp8E4M3,
    }
}

fn lower_scalar_kind(scalar: SemanticScalarTypeV1) -> Result<Type, ProductionSemanticKirErrorV1> {
    let scalar = match scalar {
        SemanticScalarTypeV1::Bool => ScalarType::Bool,
        SemanticScalarTypeV1::Integer { signed, bits } => match (signed, bits) {
            (true, 8) => ScalarType::I8,
            (true, 16) => ScalarType::I16,
            (true, 32) => ScalarType::I32,
            (true, 64) => ScalarType::I64,
            (true, 128) => ScalarType::I128,
            (false, 8) => ScalarType::U8,
            (false, 16) => ScalarType::U16,
            (false, 32) => ScalarType::U32,
            (false, 64) => ScalarType::U64,
            (false, 128) => ScalarType::U128,
            _ => {
                return Err(unsupported(
                    0,
                    None,
                    None,
                    "integer argument width is unsupported",
                ));
            }
        },
        SemanticScalarTypeV1::Float { bits } => match bits {
            16 => ScalarType::F16,
            32 => ScalarType::F32,
            64 => ScalarType::F64,
            _ => {
                return Err(unsupported(
                    0,
                    None,
                    None,
                    "floating argument width is unsupported",
                ));
            }
        },
        SemanticScalarTypeV1::Char => ScalarType::U32,
    };
    Ok(Type::Scalar(scalar))
}

fn disjoint_slice_descriptor(
    callables: &[SemanticCallableDeclV1],
    ty: SemanticTypeIdV1,
) -> Option<(SemanticTypeIdV1, SemanticTypeIdV1)> {
    let mut descriptor = None;
    for callable in callables {
        let candidate = match callable {
            SemanticCallableDeclV1::CompilerIntrinsic {
                operation:
                    SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetDisjointMut {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMutExclusive {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetBlockMut {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetTiled2dMut {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetRowStriped2dMut {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    }
                    | SemanticCompilerIntrinsicOperationV1::DisjointSliceLen {
                        disjoint_slice,
                        element,
                        raw_index,
                        ..
                    },
                ..
            } if *disjoint_slice == ty => Some((*element, *raw_index)),
            SemanticCallableDeclV1::Defined { .. }
            | SemanticCallableDeclV1::DeviceFfiImport { .. }
            | SemanticCallableDeclV1::CompilerIntrinsic { .. } => None,
        };
        if let Some(candidate) = candidate {
            if descriptor.is_some_and(|previous| previous != candidate) {
                return None;
            }
            descriptor = Some(candidate);
        }
    }
    descriptor
}

fn authenticated_workgroup_lds_scope_type_v1(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    ty: SemanticTypeIdV1,
) -> bool {
    if !callables.iter().any(|callable| {
        matches!(
            callable,
            SemanticCallableDeclV1::CompilerIntrinsic { operation, .. }
                if matches!(
                    operation,
                    SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent { scope, .. }
                        | SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate {
                            scope,
                            ..
                        } if *scope == ty
                )
        )
    }) {
        return false;
    }
    let Some(declaration) = types.get(ty.index() as usize) else {
        return false;
    };
    let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
        return false;
    };
    let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
        return false;
    };
    declaration.layout().size_bytes() == Some(0)
        && !declaration.layout().is_uninhabited()
        && matches!(
            declaration.layout().backend_repr(),
            SemanticBackendReprV1::Memory { sized: true }
        )
        && aggregate.fields().len() == layout.field_offsets().len()
        && layout.field_offsets().iter().all(|offset| *offset == 0)
        && layout.padding().is_empty()
        && aggregate.fields().iter().all(|field| {
            types.get(field.index() as usize).is_some_and(|field| {
                field.layout().size_bytes() == Some(0) && !field.layout().is_uninhabited()
            })
        })
}

#[cfg(test)]
fn disjoint_slice_operation_element(
    operation: &SemanticCompilerIntrinsicOperationV1,
    ty: SemanticTypeIdV1,
) -> Option<SemanticTypeIdV1> {
    let (disjoint_slice, element) = match operation {
        SemanticCompilerIntrinsicOperationV1::DisjointSliceLen {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetDisjointMut {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMutExclusive {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetBlockMut {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetTiled2dMut {
            disjoint_slice,
            element,
            ..
        }
        | SemanticCompilerIntrinsicOperationV1::DisjointSliceGetRowStriped2dMut {
            disjoint_slice,
            element,
            ..
        } => (*disjoint_slice, *element),
        _ => return None,
    };
    (disjoint_slice == ty).then_some(element)
}

#[cfg(test)]
mod disjoint_slice_parameter_tests {
    use super::*;

    #[test]
    fn every_disjoint_slice_intrinsic_authenticates_only_its_exact_parameter_type() {
        let disjoint_slice = SemanticTypeIdV1::from_index(1);
        let element = SemanticTypeIdV1::from_index(2);
        let witness = SemanticTypeIdV1::from_index(3);
        let raw_index = SemanticTypeIdV1::from_index(4);
        let other_slice = SemanticTypeIdV1::from_index(5);
        let index_space = SemanticDisjointIndexSpaceV1::Index1d;
        let operations = [
            SemanticCompilerIntrinsicOperationV1::DisjointSliceLen {
                disjoint_slice,
                element,
                raw_index,
                index_space,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
                disjoint_slice,
                index_witness: witness,
                element,
                raw_index,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetDisjointMut {
                disjoint_slice,
                index_witness: witness,
                element,
                raw_index,
                index_space,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMutExclusive {
                disjoint_slice,
                grid_leader: witness,
                element,
                raw_index,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetBlockMut {
                disjoint_slice,
                block_witness: witness,
                element,
                raw_index,
                index_space,
                lanes_per_block: 64,
                elements_per_lane: 4,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetTiled2dMut {
                disjoint_slice,
                tile_witness: witness,
                element,
                raw_index,
                index_space,
                lanes_per_tile: 64,
                tile_rows: 16,
                tile_columns: 16,
                elements_per_lane: 4,
            },
            SemanticCompilerIntrinsicOperationV1::DisjointSliceGetRowStriped2dMut {
                disjoint_slice,
                stripe_witness: witness,
                element,
                raw_index,
                index_space,
                lanes_per_row: 64,
                elements_per_lane: 4,
            },
        ];

        for operation in operations {
            assert_eq!(
                disjoint_slice_operation_element(&operation, disjoint_slice),
                Some(element)
            );
            assert_eq!(
                disjoint_slice_operation_element(&operation, other_slice),
                None
            );
        }
        assert_eq!(
            disjoint_slice_operation_element(
                &SemanticCompilerIntrinsicOperationV1::ThreadIndex(SemanticAxisV1::X),
                disjoint_slice,
            ),
            None
        );
    }
}

fn lower_address_space(address_space: u32) -> Result<AddressSpace, ProductionSemanticKirErrorV1> {
    match address_space {
        0 | 1 => Ok(AddressSpace::Global),
        3 => Ok(AddressSpace::Workgroup),
        4 => Ok(AddressSpace::Constant),
        5 => Ok(AddressSpace::Private),
        _ => Err(unsupported(
            0,
            None,
            None,
            "semantic pointer address space is unsupported",
        )),
    }
}

const fn lower_axis(axis: SemanticAxisV1) -> Axis {
    match axis {
        SemanticAxisV1::X => Axis::X,
        SemanticAxisV1::Y => Axis::Y,
        SemanticAxisV1::Z => Axis::Z,
    }
}

const fn inactive_launch_axis_value_v1(
    launch_rank: u8,
    kind: IndexKind,
    axis: Axis,
) -> Option<u64> {
    let axis_rank = match axis {
        Axis::X => 1,
        Axis::Y => 2,
        Axis::Z => 3,
    };
    if axis_rank <= launch_rank {
        return None;
    }
    Some(match kind {
        IndexKind::Global | IndexKind::Workgroup | IndexKind::Local => 0,
        IndexKind::WorkgroupSize | IndexKind::WorkgroupCount => 1,
    })
}

fn tiled_2d_geometry_valid(
    lanes_per_tile: u64,
    tile_rows: u64,
    tile_columns: u64,
    elements_per_lane: u64,
) -> bool {
    lanes_per_tile != 0
        && tile_rows != 0
        && tile_columns != 0
        && elements_per_lane != 0
        && lanes_per_tile.is_multiple_of(tile_columns)
        && lanes_per_tile.checked_mul(elements_per_lane) == tile_rows.checked_mul(tile_columns)
        && (lanes_per_tile / tile_columns).checked_mul(elements_per_lane) == Some(tile_rows)
}

fn row_striped_2d_geometry_valid(lanes_per_row: u64, elements_per_lane: u64) -> bool {
    lanes_per_row != 0
        && elements_per_lane != 0
        && (elements_per_lane - 1)
            .checked_mul(lanes_per_row)
            .and_then(|base| base.checked_add(lanes_per_row - 1))
            .is_some()
}

const fn lower_compare(operation: SemanticBinaryOpV1) -> Option<ComparePredicate> {
    match operation {
        SemanticBinaryOpV1::Equal => Some(ComparePredicate::Equal),
        SemanticBinaryOpV1::NotEqual => Some(ComparePredicate::NotEqual),
        SemanticBinaryOpV1::LessThan => Some(ComparePredicate::LessThan),
        SemanticBinaryOpV1::LessOrEqual => Some(ComparePredicate::LessThanOrEqual),
        SemanticBinaryOpV1::GreaterThan => Some(ComparePredicate::GreaterThan),
        SemanticBinaryOpV1::GreaterOrEqual => Some(ComparePredicate::GreaterThanOrEqual),
        SemanticBinaryOpV1::Add
        | SemanticBinaryOpV1::Subtract
        | SemanticBinaryOpV1::Multiply
        | SemanticBinaryOpV1::Divide
        | SemanticBinaryOpV1::Remainder
        | SemanticBinaryOpV1::BitXor
        | SemanticBinaryOpV1::BitAnd
        | SemanticBinaryOpV1::BitOr
        | SemanticBinaryOpV1::ShiftLeft
        | SemanticBinaryOpV1::ShiftRight
        | SemanticBinaryOpV1::Offset => None,
    }
}

const fn lower_binary(operation: SemanticBinaryOpV1) -> Option<BinaryOp> {
    match operation {
        SemanticBinaryOpV1::Add => Some(BinaryOp::Add),
        SemanticBinaryOpV1::Subtract => Some(BinaryOp::Subtract),
        SemanticBinaryOpV1::Multiply => Some(BinaryOp::Multiply),
        SemanticBinaryOpV1::Divide => Some(BinaryOp::Divide),
        SemanticBinaryOpV1::Remainder => Some(BinaryOp::Remainder),
        SemanticBinaryOpV1::BitXor => Some(BinaryOp::BitXor),
        SemanticBinaryOpV1::BitAnd => Some(BinaryOp::BitAnd),
        SemanticBinaryOpV1::BitOr => Some(BinaryOp::BitOr),
        SemanticBinaryOpV1::ShiftLeft => Some(BinaryOp::ShiftLeft),
        SemanticBinaryOpV1::ShiftRight => Some(BinaryOp::ShiftRight),
        SemanticBinaryOpV1::Equal
        | SemanticBinaryOpV1::LessThan
        | SemanticBinaryOpV1::LessOrEqual
        | SemanticBinaryOpV1::NotEqual
        | SemanticBinaryOpV1::GreaterOrEqual
        | SemanticBinaryOpV1::GreaterThan
        | SemanticBinaryOpV1::Offset => None,
    }
}

fn lower_cast_path(
    kind: SemanticCastKindV1,
    from: &Type,
    to: &Type,
) -> Option<[Option<(CastKind, ScalarType)>; 2]> {
    let (Some(from), Some(to)) = (from.as_scalar(), to.as_scalar()) else {
        return None;
    };
    if kind == SemanticCastKindV1::Integer
        && (from.is_integer() || from == ScalarType::Bool)
        && to.is_integer()
    {
        return plan_integer_cast_v1(from, to);
    }
    if from == ScalarType::Index || to == ScalarType::Index {
        return None;
    }
    let (from_width, to_width) = (from.bit_width()?, to.bit_width()?);
    let cast = match kind {
        SemanticCastKindV1::Integer if to.is_integer() => {
            if from.is_float() {
                Some(CastKind::FloatToInteger)
            } else if (from.is_integer() || from == ScalarType::Bool) && from_width > to_width {
                Some(CastKind::Truncate)
            } else if (from.is_integer() || from == ScalarType::Bool) && from_width < to_width {
                Some(if from.is_signed_integer() {
                    CastKind::SignExtend
                } else {
                    CastKind::ZeroExtend
                })
            } else {
                Some(CastKind::Bitcast)
            }
        }
        SemanticCastKindV1::Float if to.is_float() => {
            if from.is_integer() {
                Some(CastKind::IntegerToFloat)
            } else if from.is_float() && from_width < to_width {
                Some(CastKind::FloatExtend)
            } else if from.is_float() && from_width > to_width {
                Some(CastKind::FloatTruncate)
            } else {
                Some(CastKind::Bitcast)
            }
        }
        SemanticCastKindV1::Transmute if from_width == to_width => Some(CastKind::Bitcast),
        SemanticCastKindV1::Integer
        | SemanticCastKindV1::Float
        | SemanticCastKindV1::Pointer
        | SemanticCastKindV1::PointerExposeProvenance
        | SemanticCastKindV1::PointerWithExposedProvenance
        | SemanticCastKindV1::Transmute => None,
    }?;
    Some([Some((cast, to)), None])
}

fn canonical_index_constant_v1(
    operand: &SemanticOperandV1,
    lowered_type: &Type,
) -> Option<Constant> {
    if *lowered_type != Type::Scalar(ScalarType::U64) {
        return None;
    }
    let SemanticOperandV1::Constant(constant) = operand else {
        return None;
    };
    let SemanticConstantValueV1::Scalar(value) = constant.value() else {
        return None;
    };
    (value.size_bytes() == 8)
        .then(|| u64::try_from(value.bits()).ok().map(Constant::Index))
        .flatten()
}

fn canonical_shift_rhs_constant_v1(
    operation: SemanticBinaryOpV1,
    operand: &SemanticOperandV1,
    semantic_rhs_type: &Type,
    lhs_transport_type: &Type,
) -> Option<Constant> {
    if !matches!(
        operation,
        SemanticBinaryOpV1::ShiftLeft | SemanticBinaryOpV1::ShiftRight
    ) {
        return None;
    }
    let source = semantic_rhs_type.as_scalar()?;
    let destination = lhs_transport_type.as_scalar()?;
    if !source.is_integer() || !destination.is_integer() {
        return None;
    }
    let SemanticOperandV1::Constant(constant) = operand else {
        return None;
    };
    let SemanticConstantValueV1::Scalar(value) = constant.value() else {
        return None;
    };
    let source_width = match source {
        ScalarType::Index => 64,
        source => source.bit_width()?,
    };
    if u16::from(value.size_bytes()) * 8 != source_width {
        return None;
    }
    let value = value.bits();
    if source.is_signed_integer() && value & (1_u128 << (source_width - 1)) != 0 {
        return None;
    }
    match destination {
        ScalarType::I8 => i8::try_from(value).ok().map(Constant::I8),
        ScalarType::I16 => i16::try_from(value).ok().map(Constant::I16),
        ScalarType::I32 => i32::try_from(value).ok().map(Constant::I32),
        ScalarType::I64 => i64::try_from(value).ok().map(Constant::I64),
        ScalarType::U8 => u8::try_from(value).ok().map(Constant::U8),
        ScalarType::U16 => u16::try_from(value).ok().map(Constant::U16),
        ScalarType::U32 => u32::try_from(value).ok().map(Constant::U32),
        ScalarType::U64 => u64::try_from(value).ok().map(Constant::U64),
        ScalarType::Index => u64::try_from(value).ok().map(Constant::Index),
        ScalarType::Bool
        | ScalarType::I128
        | ScalarType::U128
        | ScalarType::F16
        | ScalarType::Bf16
        | ScalarType::F32
        | ScalarType::F64 => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn index_binary_coercion_v1(
    semantic_operands_match: bool,
    left: ValueId,
    left_type: &Type,
    right: ValueId,
    right_type: &Type,
) -> Option<(bool, OperationKind)> {
    if !semantic_operands_match {
        return None;
    }
    let (convert_left, value) = match (left_type, right_type) {
        (Type::Scalar(ScalarType::Index), Type::Scalar(ScalarType::U64)) => (false, right),
        (Type::Scalar(ScalarType::U64), Type::Scalar(ScalarType::Index)) => (true, left),
        (Type::Scalar(ScalarType::Index), Type::Scalar(ScalarType::U32)) => (false, right),
        (Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::Index)) => (true, left),
        _ => return None,
    };
    let kind = match (left_type, right_type) {
        (Type::Scalar(ScalarType::Index), Type::Scalar(ScalarType::U32))
        | (Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::Index)) => CastKind::ZeroExtend,
        (Type::Scalar(ScalarType::Index), Type::Scalar(ScalarType::U64))
        | (Type::Scalar(ScalarType::U64), Type::Scalar(ScalarType::Index)) => CastKind::Bitcast,
        _ => unreachable!("index coercion pair was checked above"),
    };
    Some((
        convert_left,
        OperationKind::Cast {
            kind,
            value,
            to: Type::INDEX,
        },
    ))
}

fn semantic_enum_shape(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<(SemanticTypeIdV1, &[SemanticEnumVariantV1]), ProductionSemanticKirErrorV1> {
    let declaration = types
        .get(ty.index() as usize)
        .ok_or_else(|| unsupported(0, None, None, "semantic enum type is missing"))?;
    let SemanticTypeShapeV1::Enum {
        discriminant,
        variants,
    } = declaration.shape()
    else {
        return Err(unsupported(0, None, None, "semantic value is not an enum"));
    };
    Ok((*discriminant, variants))
}

fn unique_enum_variant_with_field(
    variants: &[SemanticEnumVariantV1],
    field: SemanticTypeIdV1,
) -> Option<u32> {
    let mut matches = variants.iter().enumerate().filter_map(|(index, variant)| {
        (variant.fields().fields() == [field]).then_some(index as u32)
    });
    let found = matches.next()?;
    matches.next().is_none().then_some(found)
}

fn integer_constant(ty: &Type, bits: u128) -> Result<Constant, ProductionSemanticKirErrorV1> {
    match ty.as_scalar() {
        Some(ScalarType::Bool) if bits <= 1 => Ok(Constant::Bool(bits != 0)),
        Some(ScalarType::I8) => Ok(Constant::I8(bits as u8 as i8)),
        Some(ScalarType::I16) => Ok(Constant::I16(bits as u16 as i16)),
        Some(ScalarType::I32) => Ok(Constant::I32(bits as u32 as i32)),
        Some(ScalarType::I64) => Ok(Constant::I64(bits as u64 as i64)),
        Some(ScalarType::U8) => Ok(Constant::U8(bits as u8)),
        Some(ScalarType::U16) => Ok(Constant::U16(bits as u16)),
        Some(ScalarType::U32) => Ok(Constant::U32(bits as u32)),
        Some(ScalarType::U64) => Ok(Constant::U64(bits as u64)),
        Some(ScalarType::Index) => Ok(Constant::Index(bits as u64)),
        Some(ScalarType::I128 | ScalarType::U128) => Err(unsupported(
            0,
            None,
            None,
            "128-bit enum discriminants have no Kernel IR V1 representation",
        )),
        Some(_) | None => Err(unsupported(
            0,
            None,
            None,
            "semantic enum discriminant has no integer Kernel IR representation",
        )),
    }
}

fn checked_constant_range_v1(bytes: &[u8], offset: u64, size: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let size = usize::try_from(size).ok()?;
    let end = start.checked_add(size)?;
    bytes.get(start..end)
}

fn read_constant_bits_v1(bytes: &[u8], offset: u64, size: u64) -> Option<u128> {
    if size == 0 || size > 16 {
        return None;
    }
    let bytes = checked_constant_range_v1(bytes, offset, size)?;
    Some(
        bytes
            .iter()
            .enumerate()
            .fold(0_u128, |bits, (index, byte)| {
                bits | (u128::from(*byte) << (index * 8))
            }),
    )
}

fn lower_constant(
    ty: Type,
    value: SemanticScalarValueV1,
) -> Result<Constant, ProductionSemanticKirErrorV1> {
    let bits = value.bits();
    let constant = match ty.as_scalar() {
        Some(ScalarType::Bool) if value.size_bytes() == 1 && bits <= 1 => Constant::Bool(bits != 0),
        Some(ScalarType::I8) if value.size_bytes() == 1 => Constant::I8(bits as u8 as i8),
        Some(ScalarType::I16) if value.size_bytes() == 2 => Constant::I16(bits as u16 as i16),
        Some(ScalarType::I32) if value.size_bytes() == 4 => Constant::I32(bits as u32 as i32),
        Some(ScalarType::I64) if value.size_bytes() == 8 => Constant::I64(bits as u64 as i64),
        Some(ScalarType::U8) if value.size_bytes() == 1 => Constant::U8(bits as u8),
        Some(ScalarType::U16) if value.size_bytes() == 2 => Constant::U16(bits as u16),
        Some(ScalarType::U32) if value.size_bytes() == 4 => Constant::U32(bits as u32),
        Some(ScalarType::U64) if value.size_bytes() == 8 => Constant::U64(bits as u64),
        Some(ScalarType::Index) if value.size_bytes() == 8 => Constant::Index(bits as u64),
        Some(ScalarType::F16) if value.size_bytes() == 2 => Constant::F16Bits(bits as u16),
        Some(ScalarType::F32) if value.size_bytes() == 4 => Constant::F32Bits(bits as u32),
        Some(ScalarType::F64) if value.size_bytes() == 8 => Constant::F64Bits(bits as u64),
        Some(ScalarType::Bf16) if value.size_bytes() == 2 => Constant::Bf16Bits(bits as u16),
        Some(ScalarType::I128 | ScalarType::U128) => {
            return Err(unsupported(
                0,
                None,
                None,
                "128-bit constants have no Kernel IR V1 representation",
            ));
        }
        Some(_) | None => {
            return Err(unsupported(
                0,
                None,
                None,
                "semantic scalar constant size does not match its lowered type",
            ));
        }
    };
    Ok(constant)
}

fn memory_access_for_type(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    address_space: AddressSpace,
) -> Result<MemoryAccess, ProductionSemanticKirErrorV1> {
    let alignment = types
        .get(usize::try_from(ty.index()).unwrap_or(usize::MAX))
        .ok_or_else(|| unsupported(0, None, None, "memory access type is missing"))?
        .layout()
        .alignment_bytes();
    let alignment = u32::try_from(alignment)
        .ok()
        .filter(|alignment| *alignment != 0)
        .ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "memory access alignment has no Kernel IR V1 representation",
            )
        })?;
    Ok(MemoryAccess::new(address_space, alignment))
}

fn enforce_limit(
    resource: ProductionSemanticKirResourceV1,
    actual: usize,
    limit: usize,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if actual > limit {
        Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn authenticated_subgroup_broadcast_source_is_bounded(exclusive_bound: u128, width: u32) -> bool {
    width != 0 && width.is_power_of_two() && width <= 64 && exclusive_bound <= u128::from(width)
}

fn subgroup_broadcast_source_is_statically_bounded(
    operations: &[Operation],
    source_lane: ValueId,
    width: u32,
) -> bool {
    if width == 0 || !width.is_power_of_two() || width > 64 {
        return false;
    }
    let u32_constant = |value| {
        operations.iter().find_map(|operation| {
            let defines_value = operation
                .results
                .iter()
                .any(|result| result.id == value && result.ty == Type::Scalar(ScalarType::U32));
            match operation.kind {
                OperationKind::Constant(Constant::U32(constant)) if defines_value => Some(constant),
                _ => None,
            }
        })
    };
    let Some(producer) = operations.iter().find(|operation| {
        operation
            .results
            .iter()
            .any(|result| result.id == source_lane && result.ty == Type::Scalar(ScalarType::U32))
    }) else {
        return false;
    };
    match producer.kind {
        OperationKind::Constant(Constant::U32(lane)) => lane < width,
        OperationKind::Binary {
            op: BinaryOp::BitAnd,
            lhs,
            rhs,
        } => {
            u32_constant(lhs).is_some_and(|mask| mask < width)
                || u32_constant(rhs).is_some_and(|mask| mask < width)
        }
        _ => false,
    }
}

const fn unsupported(
    function: u32,
    block: Option<u32>,
    statement: Option<u32>,
    detail: &'static str,
) -> ProductionSemanticKirErrorV1 {
    ProductionSemanticKirErrorV1::Unsupported {
        function,
        block,
        statement,
        detail,
    }
}

fn hex_identity(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    use dialect_kernel::AccessKindAttr;
    use fe2o3_mir_model::semantic_mir_v1::{
        SemanticAbiArgumentV1, SemanticAbiIdentityV1, SemanticAbiPassModeV1, SemanticAbiValueV1,
        SemanticAggregateLayoutV1, SemanticAggregateTypeV1, SemanticAssignmentV1,
        SemanticBackendReprV1, SemanticBasicBlockV1, SemanticBlockIdentityV1,
        SemanticCallDestinationV1, SemanticCallableIdV1, SemanticCanonAbiV1,
        SemanticCompilerIntrinsicIdentityV1, SemanticConstGenericArgumentsIdentityV1,
        SemanticConstantV1, SemanticControlFlowEdgeV1, SemanticEdgeRoleV1, SemanticExternAbiV1,
        SemanticFieldsShapeV1, SemanticFunctionAbiV1, SemanticFunctionIdentityV1,
        SemanticFunctionRoleV1, SemanticGenericTypeArgumentsIdentityV1,
        SemanticItemDefinitionIdentityV1, SemanticLayoutIdentityV1, SemanticLocalDeclV1,
        SemanticLocalIdV1, SemanticLocalIdentityV1, SemanticMfmaAccumulatorDistributionV1,
        SemanticMfmaOperandRoleV1, SemanticMfmaRegisterDistributionV1,
        SemanticMonomorphizationIdentityV1, SemanticNonBodyCallableBindingV1, SemanticProjectionV1,
        SemanticRustcVariantsV1, SemanticRvalueV1, SemanticSourceProvenanceV1, SemanticStatementV1,
        SemanticSwitchTargetV1, SemanticSwitchTargetsV1, SemanticTerminatorV1,
        SemanticTypeIdentityV1, SemanticTypeLayoutDetailsV1, SemanticTypeLayoutV1,
    };
    use fe2o3_pliron::{
        ProductionConstructionV1, ProductionRankedBlockV1, ProductionRankedKernelV1,
        ProductionRankedTerminatorV1, ProductionRankedValueIdV1, ProductionSessionLimitsV1,
        compile_ranked_kernel_for_gfx942_lowering_v1, compile_ranked_kernel_for_lowering_v1,
    };

    #[test]
    fn subgroup_broadcast_accepts_local_u32_mask_below_wave64_width() {
        let unknown = ValueId(0);
        let mask = ValueId(1);
        let masked = ValueId(2);
        let constant = Operation::effect_free(
            ValueDef::new(mask, Type::Scalar(ScalarType::U32)),
            OperationKind::Constant(Constant::U32(63)),
        );
        for kind in [
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs: unknown,
                rhs: mask,
            },
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs: mask,
                rhs: unknown,
            },
        ] {
            let operations = [
                constant.clone(),
                Operation::effect_free(ValueDef::new(masked, Type::Scalar(ScalarType::U32)), kind),
            ];
            assert!(subgroup_broadcast_source_is_statically_bounded(
                &operations,
                masked,
                64,
            ));
        }
        assert!(subgroup_broadcast_source_is_statically_bounded(
            &[constant],
            mask,
            64,
        ));
    }

    #[test]
    fn subgroup_broadcast_rejects_mask_at_width_and_missing_constant_mask() {
        let unknown = ValueId(0);
        let mask = ValueId(1);
        let masked = ValueId(2);
        let mask64 = Operation::effect_free(
            ValueDef::new(mask, Type::Scalar(ScalarType::U32)),
            OperationKind::Constant(Constant::U32(64)),
        );
        let bitand = |rhs| {
            Operation::effect_free(
                ValueDef::new(masked, Type::Scalar(ScalarType::U32)),
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: unknown,
                    rhs,
                },
            )
        };
        assert!(!subgroup_broadcast_source_is_statically_bounded(
            &[mask64, bitand(mask)],
            masked,
            64,
        ));
        assert!(!subgroup_broadcast_source_is_statically_bounded(
            &[bitand(ValueId(3))],
            masked,
            64,
        ));
    }

    #[derive(Clone, Copy)]
    struct AuthenticatedInductionFixtureV1 {
        bits: u16,
        bound: u128,
        step: u128,
        extra_write: bool,
        bypass_guard: bool,
    }

    impl Default for AuthenticatedInductionFixtureV1 {
        fn default() -> Self {
            Self {
                bits: 64,
                bound: 64,
                step: 1,
                extra_write: false,
                bypass_guard: false,
            }
        }
    }

    fn authenticated_induction_fixture_v1(
        options: AuthenticatedInductionFixtureV1,
    ) -> (Vec<SemanticTypeDeclV1>, SemanticFunctionDeclV1) {
        let unit = SemanticTypeIdV1::from_index(0);
        let induction_ty = SemanticTypeIdV1::from_index(1);
        let bool_ty = SemanticTypeIdV1::from_index(2);
        let u32_ty = SemanticTypeIdV1::from_index(3);
        let source = SemanticSourceProvenanceV1::unavailable();
        let size = u8::try_from(options.bits / 8).unwrap();
        let place = |local, ty| {
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap()
        };
        let operand = |local, ty| SemanticOperandV1::Copy(place(local, ty));
        let constant = |ty, value| {
            SemanticOperandV1::Constant(SemanticConstantV1::new(
                ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, size).unwrap()),
            ))
        };
        let assign = |local, ty, value| {
            SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                    place(local, ty),
                    SemanticRvalueV1::new(ty, value),
                )),
            )
        };
        let edge = |role, target| {
            SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target))
        };
        let block = |tag, statements, terminator| {
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([tag; 32]),
                source,
                statements,
                SemanticTerminatorV1::new(source, terminator),
            )
            .unwrap()
        };
        let entry = block(
            210,
            vec![assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Use(constant(induction_ty, 0)),
            )],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
        );
        let header = block(
            211,
            vec![assign(
                2,
                bool_ty,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::LessThan,
                    left: operand(1, induction_ty),
                    right: constant(induction_ty, options.bound),
                },
            )],
            SemanticTerminatorKindV1::SwitchInt {
                discriminant: operand(2, bool_ty),
                targets: SemanticSwitchTargetsV1::new(
                    vec![SemanticSwitchTargetV1::new(
                        0,
                        edge(SemanticEdgeRoleV1::SwitchValue, 4),
                    )],
                    edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                )
                .unwrap(),
            },
        );
        let mut body_statements = vec![assign(
            3,
            u32_ty,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Integer,
                operand: operand(1, induction_ty),
            },
        )];
        if options.extra_write {
            body_statements.push(assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Use(constant(induction_ty, 0)),
            ));
        }
        let body = block(
            212,
            body_statements,
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 3)),
        );
        let latch = block(
            213,
            vec![assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::Add,
                    left: operand(1, induction_ty),
                    right: constant(induction_ty, options.step),
                },
            )],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
        );
        let exit = block(
            214,
            vec![],
            if options.bypass_guard {
                SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 2))
            } else {
                SemanticTerminatorKindV1::Return
            },
        );
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([215; 32]),
            SemanticLayoutIdentityV1::from_sha256([216; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let locals: Vec<SemanticLocalDeclV1> = [
            (unit, SemanticLocalRoleV1::Return),
            (induction_ty, SemanticLocalRoleV1::Temporary),
            (bool_ty, SemanticLocalRoleV1::Temporary),
            (u32_ty, SemanticLocalRoleV1::Temporary),
        ]
        .into_iter()
        .enumerate()
        .map(|(local, (ty, role))| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([217 + local as u8; 32]),
                ty,
                role,
                source,
            )
        })
        .collect();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([221; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([222; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([223; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([224; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([225; 32]),
            source,
            abi,
            locals,
            SemanticBlockIdV1::from_index(0),
            vec![entry, header, body, latch, exit],
        )
        .unwrap();
        (
            vec![
                unit_type(),
                unsigned_scalar_type(226, options.bits),
                bool_type(),
                unsigned_scalar_type(228, 32),
            ],
            function,
        )
    }

    #[test]
    fn ranked_canonical_induction_bound_survives_exact_u64_to_u32_cast() {
        let (types, function) =
            authenticated_induction_fixture_v1(AuthenticatedInductionFixtureV1::default());
        let bounds = authenticated_loop_induction_bounds_v1(&types, &function).unwrap();
        assert_eq!(bounds.get(&(2, 1)), Some(&64));
        let alias = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(3),
                vec![],
                SemanticTypeIdV1::from_index(3),
            )
            .unwrap(),
        );
        assert_eq!(
            authenticated_unsigned_operand_exclusive_bound_v1(
                &types,
                &function,
                &bounds,
                SemanticBlockIdV1::from_index(2),
                &alias,
            ),
            Some(64),
        );
    }

    #[test]
    fn authenticated_induction_bound_rejects_width_overrun_and_hostile_loops() {
        let (types, function) =
            authenticated_induction_fixture_v1(AuthenticatedInductionFixtureV1 {
                bound: 65,
                ..AuthenticatedInductionFixtureV1::default()
            });
        let bounds = authenticated_loop_induction_bounds_v1(&types, &function).unwrap();
        assert_eq!(bounds.get(&(2, 1)), Some(&65));
        assert!(!authenticated_subgroup_broadcast_source_is_bounded(
            *bounds.get(&(2, 1)).unwrap(),
            64,
        ));
        assert!(!authenticated_subgroup_broadcast_source_is_bounded(0, 0));

        for options in [
            AuthenticatedInductionFixtureV1 {
                extra_write: true,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                step: 0,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                bits: 8,
                bound: 255,
                step: 2,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                bypass_guard: true,
                ..AuthenticatedInductionFixtureV1::default()
            },
        ] {
            let (types, function) = authenticated_induction_fixture_v1(options);
            assert!(
                authenticated_loop_induction_bounds_v1(&types, &function)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn inactive_launch_axes_use_canonical_identity_and_extent_constants() {
        for rank in 1..=3 {
            for (axis, axis_rank) in [(Axis::X, 1), (Axis::Y, 2), (Axis::Z, 3)] {
                for kind in [IndexKind::Global, IndexKind::Workgroup, IndexKind::Local] {
                    assert_eq!(
                        inactive_launch_axis_value_v1(rank, kind, axis),
                        (axis_rank > rank).then_some(0),
                    );
                }
                for kind in [IndexKind::WorkgroupSize, IndexKind::WorkgroupCount] {
                    assert_eq!(
                        inactive_launch_axis_value_v1(rank, kind, axis),
                        (axis_rank > rank).then_some(1),
                    );
                }
            }
        }
    }

    #[test]
    fn semantic_casts_use_the_shared_bounded_kernel_ir_index_paths() {
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U32),
                &Type::INDEX,
            ),
            Some([Some((CastKind::ZeroExtend, ScalarType::Index)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U64),
                &Type::INDEX,
            ),
            Some([Some((CastKind::Bitcast, ScalarType::Index)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::U64),
            ),
            Some([Some((CastKind::Bitcast, ScalarType::U64)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::I32),
                &Type::INDEX,
            ),
            Some([
                Some((CastKind::SignExtend, ScalarType::U64)),
                Some((CastKind::Bitcast, ScalarType::Index)),
            ])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::U32),
            ),
            Some([
                Some((CastKind::Bitcast, ScalarType::U64)),
                Some((CastKind::Truncate, ScalarType::U32)),
            ])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::F64),
            ),
            None
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Float,
                &Type::Scalar(ScalarType::U64),
                &Type::INDEX,
            ),
            None
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U32),
                &Type::Scalar(ScalarType::U64),
            ),
            Some([Some((CastKind::ZeroExtend, ScalarType::U64)), None])
        );
    }

    #[test]
    fn admitted_strided_read_scalars_have_exact_byte_alignments() {
        for (scalar, alignment) in [
            (ScalarType::Bool, 1),
            (ScalarType::I8, 1),
            (ScalarType::U16, 2),
            (ScalarType::F32, 4),
            (ScalarType::I64, 8),
            (ScalarType::F64, 8),
        ] {
            assert_eq!(
                strided_read_scalar_alignment_v1(&Type::Scalar(scalar)),
                Some(alignment)
            );
        }
        assert_eq!(strided_read_scalar_alignment_v1(&Type::INDEX), None);
    }

    #[test]
    fn matched_usize_constants_are_recognized_in_both_operand_orders() {
        let semantic_type = SemanticTypeIdV1::from_index(0);
        let constant = SemanticOperandV1::Constant(SemanticConstantV1::new(
            semantic_type,
            SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(64, 8).unwrap()),
        ));
        let u64_type = Type::Scalar(ScalarType::U64);

        assert_eq!(
            canonical_index_constant_v1(&constant, &u64_type),
            Some(Constant::Index(64))
        );
        let left = index_binary_coercion_v1(true, ValueId(1), &u64_type, ValueId(2), &Type::INDEX)
            .unwrap();
        assert!(left.0);
        assert!(matches!(
            left.1,
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value: ValueId(1),
                to: Type::Scalar(ScalarType::Index),
            }
        ));

        let right = index_binary_coercion_v1(true, ValueId(2), &Type::INDEX, ValueId(1), &u64_type)
            .unwrap();
        assert!(!right.0);
        assert!(matches!(
            right.1,
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value: ValueId(1),
                to: Type::Scalar(ScalarType::Index),
            }
        ));
    }

    #[test]
    fn matching_u32_values_coerce_to_an_authenticated_index_transport() {
        let index = Type::INDEX;
        let u32_type = Type::Scalar(ScalarType::U32);
        let i32_type = Type::Scalar(ScalarType::I32);

        assert_eq!(
            index_binary_coercion_v1(true, ValueId(1), &index, ValueId(2), &u32_type),
            Some((
                false,
                OperationKind::Cast {
                    kind: CastKind::ZeroExtend,
                    value: ValueId(2),
                    to: Type::INDEX,
                },
            ))
        );
        assert_eq!(
            index_binary_coercion_v1(true, ValueId(1), &u32_type, ValueId(2), &index),
            Some((
                true,
                OperationKind::Cast {
                    kind: CastKind::ZeroExtend,
                    value: ValueId(1),
                    to: Type::INDEX,
                },
            ))
        );
        assert!(
            index_binary_coercion_v1(true, ValueId(1), &index, ValueId(2), &i32_type).is_none()
        );
        assert!(
            index_binary_coercion_v1(false, ValueId(1), &index, ValueId(2), &u32_type).is_none()
        );
    }

    #[test]
    fn binary_lowering_emits_one_canonical_index_constant_without_dead_u64_ir() {
        fn lower(
            constant_on_left: bool,
            max_operations: usize,
        ) -> Result<Vec<Operation>, ProductionSemanticKirErrorV1> {
            let unit = SemanticTypeIdV1::from_index(0);
            let u64_ty = SemanticTypeIdV1::from_index(1);
            let types = [unit_type(), u64_type()];
            let source = SemanticSourceProvenanceV1::unavailable();
            let abi = SemanticFunctionAbiV1::from_rustc(
                SemanticAbiIdentityV1::from_sha256([21; 32]),
                SemanticLayoutIdentityV1::from_sha256([22; 32]),
                SemanticCanonAbiV1::GpuKernel,
                SemanticExternAbiV1::GpuKernel,
                false,
                false,
                0,
                vec![],
                SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
            )
            .unwrap();
            let block = SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([23; 32]),
                source,
                vec![],
                SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
            )
            .unwrap();
            let function = SemanticFunctionDeclV1::new(
                SemanticFunctionIdentityV1::from_sha256([24; 32]),
                SemanticFunctionRoleV1::InternalHelper,
                SemanticItemDefinitionIdentityV1::from_sha256([25; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([26; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([27; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([28; 32]),
                source,
                abi,
                vec![
                    SemanticLocalDeclV1::new(
                        SemanticLocalIdentityV1::from_sha256([29; 32]),
                        unit,
                        SemanticLocalRoleV1::Return,
                        source,
                    ),
                    SemanticLocalDeclV1::new(
                        SemanticLocalIdentityV1::from_sha256([30; 32]),
                        u64_ty,
                        SemanticLocalRoleV1::Temporary,
                        source,
                    ),
                ],
                SemanticBlockIdV1::from_index(0),
                vec![block],
            )
            .unwrap();
            let mut lowering = SemanticFunctionLoweringV1::new(
                &types,
                &[],
                &function,
                SemanticParameterBindingsV1 {
                    declarations: &[],
                    values: &[],
                    types: &[],
                },
                None,
                None,
                BTreeSet::new(),
                1,
                false,
                max_operations,
            )
            .unwrap();
            lowering.locals[1] = Some(SemanticValueBindingV1::Value {
                id: ValueId(0),
                ty: Type::INDEX,
            });
            lowering.next_value = 1;
            let constant = SemanticOperandV1::Constant(SemanticConstantV1::new(
                u64_ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(64, 8).unwrap()),
            ));
            let dynamic = SemanticOperandV1::Copy(
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(1), vec![], u64_ty).unwrap(),
            );
            let (left, right) = if constant_on_left {
                (constant, dynamic)
            } else {
                (dynamic, constant)
            };
            let mut operations = Vec::new();
            lowering.lower_rvalue(
                SemanticBlockIdV1::from_index(0),
                Some(0),
                u64_ty,
                &SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::Add,
                    left,
                    right,
                },
                &mut operations,
            )?;
            Ok(operations)
        }

        for operations in [lower(true, 2).unwrap(), lower(false, 2).unwrap()] {
            assert_eq!(operations.len(), 2);
            assert_eq!(
                operations
                    .iter()
                    .filter(|operation| matches!(
                        &operation.kind,
                        OperationKind::Constant(Constant::Index(64))
                    ))
                    .count(),
                1
            );
            assert!(!operations.iter().any(|operation| matches!(
                &operation.kind,
                OperationKind::Constant(Constant::U64(64)) | OperationKind::Cast { .. }
            )));
            assert!(matches!(
                &operations[1].kind,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
        }
        assert!(matches!(
            lower(true, 1),
            Err(ProductionSemanticKirErrorV1::ResourceLimit {
                resource: ProductionSemanticKirResourceV1::Operations,
                actual: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn index_binary_coercion_keeps_dynamic_values_and_mismatches_fail_closed() {
        let u64_type = Type::Scalar(ScalarType::U64);
        let dynamic_coercion =
            index_binary_coercion_v1(true, ValueId(1), &Type::INDEX, ValueId(2), &u64_type)
                .unwrap();
        assert!(matches!(
            dynamic_coercion,
            (
                false,
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: ValueId(2),
                    to: Type::Scalar(ScalarType::Index),
                }
            )
        ));
        assert!(
            index_binary_coercion_v1(false, ValueId(1), &Type::INDEX, ValueId(2), &u64_type,)
                .is_none()
        );
    }

    fn lower_heterogeneous_u8_shift(
        right: SemanticOperandV1,
        max_operations: usize,
    ) -> Result<Vec<Operation>, ProductionSemanticKirErrorV1> {
        let unit = SemanticTypeIdV1::from_index(0);
        let u8_ty = SemanticTypeIdV1::from_index(1);
        let i32_ty = SemanticTypeIdV1::from_index(2);
        let types = [
            unit_type(),
            integer_type(81, false, 8),
            integer_type(83, true, 32),
        ];
        let source = SemanticSourceProvenanceV1::unavailable();
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([85; 32]),
            SemanticLayoutIdentityV1::from_sha256([86; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let block = SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([87; 32]),
            source,
            vec![],
            SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([88; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([89; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([90; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([91; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([92; 32]),
            source,
            abi,
            vec![
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([93; 32]),
                    unit,
                    SemanticLocalRoleV1::Return,
                    source,
                ),
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([94; 32]),
                    u8_ty,
                    SemanticLocalRoleV1::Temporary,
                    source,
                ),
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([95; 32]),
                    i32_ty,
                    SemanticLocalRoleV1::Temporary,
                    source,
                ),
            ],
            SemanticBlockIdV1::from_index(0),
            vec![block],
        )
        .unwrap();
        let mut lowering = SemanticFunctionLoweringV1::new(
            &types,
            &[],
            &function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            max_operations,
        )
        .unwrap();
        lowering.locals[1] = Some(SemanticValueBindingV1::Value {
            id: ValueId(0),
            ty: Type::Scalar(ScalarType::U8),
        });
        lowering.locals[2] = Some(SemanticValueBindingV1::Value {
            id: ValueId(1),
            ty: Type::Scalar(ScalarType::I32),
        });
        lowering.next_value = 2;
        let left = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(1), vec![], u8_ty).unwrap(),
        );
        let mut operations = Vec::new();
        lowering.lower_rvalue(
            SemanticBlockIdV1::from_index(0),
            Some(0),
            u8_ty,
            &SemanticRvalueKindV1::Binary {
                operation: SemanticBinaryOpV1::ShiftRight,
                left,
                right,
            },
            &mut operations,
        )?;
        Ok(operations)
    }

    #[test]
    fn heterogeneous_constant_shift_count_is_canonicalized_to_the_lhs_type() {
        let i32_ty = SemanticTypeIdV1::from_index(2);
        let right = SemanticOperandV1::Constant(SemanticConstantV1::new(
            i32_ty,
            SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(3, 4).unwrap()),
        ));
        let operations = lower_heterogeneous_u8_shift(right, 2).unwrap();
        assert_eq!(operations.len(), 2);
        assert!(matches!(
            &operations[0].kind,
            OperationKind::Constant(Constant::U8(3))
        ));
        assert!(matches!(
            &operations[1],
            Operation {
                results,
                kind: OperationKind::Binary {
                    op: BinaryOp::ShiftRight,
                    lhs: ValueId(0),
                    rhs: ValueId(2),
                },
                ..
            } if results == &[ValueDef::new(ValueId(3), Type::Scalar(ScalarType::U8))]
        ));
        assert!(!operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Constant(Constant::I32(_)) | OperationKind::Cast { .. }
        )));
    }

    #[test]
    fn heterogeneous_shift_count_rejects_nonconstant_and_unrepresentable_values() {
        let i32_ty = SemanticTypeIdV1::from_index(2);
        let dynamic = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(2), vec![], i32_ty).unwrap(),
        );
        assert!(matches!(
            lower_heterogeneous_u8_shift(dynamic, 2),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "semantic binary operand types differ",
                ..
            })
        ));

        let constant = |bits| {
            SemanticOperandV1::Constant(SemanticConstantV1::new(
                i32_ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(bits, 4).unwrap()),
            ))
        };
        for hostile in [constant(u128::from(u32::MAX)), constant(256)] {
            assert!(matches!(
                lower_heterogeneous_u8_shift(hostile, 2),
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "semantic binary operand types differ",
                    ..
                })
            ));
        }

        let float = SemanticOperandV1::Constant(SemanticConstantV1::new(
            i32_ty,
            SemanticConstantValueV1::Scalar(
                SemanticScalarValueV1::new(u128::from(3.0_f32.to_bits()), 4).unwrap(),
            ),
        ));
        assert!(
            canonical_shift_rhs_constant_v1(
                SemanticBinaryOpV1::ShiftRight,
                &float,
                &Type::Scalar(ScalarType::F32),
                &Type::Scalar(ScalarType::U8),
            )
            .is_none()
        );
        assert!(
            canonical_shift_rhs_constant_v1(
                SemanticBinaryOpV1::Add,
                &constant(3),
                &Type::Scalar(ScalarType::I32),
                &Type::Scalar(ScalarType::U8),
            )
            .is_none()
        );
    }

    fn unit_type() -> SemanticTypeDeclV1 {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([1; 32]),
            SemanticLayoutIdentityV1::from_sha256([2; 32]),
            SemanticTypeLayoutV1::with_exact_rustc_layout(
                0,
                1,
                SemanticFieldsShapeV1::arbitrary(vec![], vec![]).unwrap(),
                SemanticRustcVariantsV1::Single { index: 0 },
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                1,
                0,
                SemanticTypeLayoutDetailsV1::None,
            )
            .unwrap(),
            SemanticTypeShapeV1::Unit,
        )
    }

    fn u64_type() -> SemanticTypeDeclV1 {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([31; 32]),
            SemanticLayoutIdentityV1::from_sha256([32; 32]),
            SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 64,
            }),
        )
    }

    fn integer_type(identity: u8, signed: bool, bits: u16) -> SemanticTypeDeclV1 {
        let size = u64::from(bits / 8);
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([identity; 32]),
            SemanticLayoutIdentityV1::from_sha256([identity.wrapping_add(1); 32]),
            SemanticTypeLayoutV1::new(Some(size), size).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer { signed, bits }),
        )
    }

    fn unsigned_scalar_type(tag: u8, bits: u16) -> SemanticTypeDeclV1 {
        let bytes = u64::from(bits / 8);
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag.wrapping_add(1); 32]),
            SemanticTypeLayoutV1::new(Some(bytes), bytes).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits,
            }),
        )
    }

    fn bool_type() -> SemanticTypeDeclV1 {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([35; 32]),
            SemanticLayoutIdentityV1::from_sha256([36; 32]),
            SemanticTypeLayoutV1::new(Some(1), 1).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool),
        )
    }

    fn lower_bf16_conversion_for_test(
        kind: SemanticBf16ConversionKindV1,
    ) -> (Vec<Operation>, SemanticValueBindingV1) {
        let unit = SemanticTypeIdV1::from_index(0);
        let u16_ty = SemanticTypeIdV1::from_index(1);
        let bf16_ty = SemanticTypeIdV1::from_index(2);
        let f32_ty = SemanticTypeIdV1::from_index(3);
        let types = vec![
            unit_type(),
            unsigned_scalar_type(101, 16),
            SemanticTypeDeclV1::new(
                SemanticTypeIdentityV1::from_sha256([103; 32]),
                SemanticLayoutIdentityV1::from_sha256([104; 32]),
                SemanticTypeLayoutV1::aggregate(
                    Some(2),
                    2,
                    SemanticAggregateLayoutV1::new(vec![0], vec![]).unwrap(),
                )
                .unwrap(),
                SemanticTypeShapeV1::Aggregate(SemanticAggregateTypeV1::new(vec![u16_ty]).unwrap()),
            ),
            SemanticTypeDeclV1::new(
                SemanticTypeIdentityV1::from_sha256([105; 32]),
                SemanticLayoutIdentityV1::from_sha256([106; 32]),
                SemanticTypeLayoutV1::new(Some(4), 4).unwrap(),
                SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits: 32 }),
            ),
        ];
        let (input, output, input_binding) = match kind {
            SemanticBf16ConversionKindV1::FromBits => (
                u16_ty,
                bf16_ty,
                SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: Type::Scalar(ScalarType::U16),
                },
            ),
            SemanticBf16ConversionKindV1::ToBits => (
                bf16_ty,
                u16_ty,
                SemanticValueBindingV1::Aggregate(vec![SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: Type::Scalar(ScalarType::U16),
                }]),
            ),
            SemanticBf16ConversionKindV1::FromF32RoundTiesEven => (
                f32_ty,
                bf16_ty,
                SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: Type::Scalar(ScalarType::F32),
                },
            ),
            SemanticBf16ConversionKindV1::ToF32 => (
                bf16_ty,
                f32_ty,
                SemanticValueBindingV1::Aggregate(vec![SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: Type::Scalar(ScalarType::U16),
                }]),
            ),
        };
        let source = SemanticSourceProvenanceV1::unavailable();
        let value = |ty| {
            SemanticAbiValueV1::new(
                ty,
                SemanticAbiPassModeV1::Direct(
                    fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                ),
            )
        };
        let intrinsic_abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([107; 32]),
            SemanticLayoutIdentityV1::from_sha256([108; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            1,
            vec![SemanticAbiArgumentV1::source(value(input))],
            value(output),
        )
        .unwrap();
        let callable = SemanticCallableDeclV1::CompilerIntrinsic {
            binding: SemanticNonBodyCallableBindingV1::new(
                SemanticFunctionIdentityV1::from_sha256([109; 32]),
                SemanticItemDefinitionIdentityV1::from_sha256([110; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([111; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([112; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([113; 32]),
                source,
                intrinsic_abi,
            ),
            operation: SemanticCompilerIntrinsicOperationV1::Bf16Conversion {
                kind,
                input,
                output,
            },
            operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([114; 32]),
        };
        let function_abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([115; 32]),
            SemanticLayoutIdentityV1::from_sha256([116; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let place = |local, ty| {
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap()
        };
        let block = SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([117; 32]),
            source,
            vec![],
            SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([118; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([119; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([120; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([121; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([122; 32]),
            source,
            function_abi,
            vec![
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([123; 32]),
                    unit,
                    SemanticLocalRoleV1::Return,
                    source,
                ),
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([124; 32]),
                    input,
                    SemanticLocalRoleV1::Temporary,
                    source,
                ),
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([125; 32]),
                    output,
                    SemanticLocalRoleV1::Temporary,
                    source,
                ),
            ],
            SemanticBlockIdV1::from_index(0),
            vec![block],
        )
        .unwrap();
        let callables = [callable];
        let mut lowering = SemanticFunctionLoweringV1::new(
            &types,
            &callables,
            &function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            16,
        )
        .unwrap();
        lowering.locals[1] = Some(input_binding);
        lowering.next_value = 8;
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![SemanticOperandV1::Copy(place(1, input))],
            Some(SemanticCallDestinationV1::new(
                place(2, output),
                SemanticControlFlowEdgeV1::new(
                    SemanticEdgeRoleV1::CallReturn,
                    SemanticBlockIdV1::from_index(0),
                ),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let mut operations = Vec::new();
        lowering
            .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
            .unwrap();
        (operations, lowering.locals[2].clone().unwrap())
    }

    #[test]
    fn bf16_conversions_lower_to_exact_verified_kernel_ir_sequences() {
        let (from_bits, from_bits_binding) =
            lower_bf16_conversion_for_test(SemanticBf16ConversionKindV1::FromBits);
        assert!(from_bits.is_empty());
        assert_eq!(
            from_bits_binding.values().unwrap(),
            vec![(ValueId(7), Type::Scalar(ScalarType::U16))]
        );

        let (to_bits, to_bits_binding) =
            lower_bf16_conversion_for_test(SemanticBf16ConversionKindV1::ToBits);
        assert!(to_bits.is_empty());
        assert_eq!(
            to_bits_binding.value().unwrap(),
            (ValueId(7), Type::Scalar(ScalarType::U16))
        );

        let (from_f32, from_f32_binding) =
            lower_bf16_conversion_for_test(SemanticBf16ConversionKindV1::FromF32RoundTiesEven);
        assert_eq!(from_f32.len(), 2);
        assert!(matches!(
            &from_f32[0].kind,
            OperationKind::Call { callee, arguments }
                if *callee == FunctionId::new("__fe2o3_ir_float_v1_f32_to_bf16_rne")
                    && arguments == &[ValueId(7)]
        ));
        assert!(matches!(
            from_f32[1].kind,
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                ..
            }
        ));
        assert_eq!(
            from_f32_binding.values().unwrap()[0].1,
            Type::Scalar(ScalarType::U16)
        );

        let (to_f32, to_f32_binding) =
            lower_bf16_conversion_for_test(SemanticBf16ConversionKindV1::ToF32);
        assert_eq!(to_f32.len(), 2);
        assert!(matches!(
            to_f32[0].kind,
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                ..
            }
        ));
        assert!(matches!(
            &to_f32[1].kind,
            OperationKind::Call { callee, arguments }
                if *callee == FunctionId::new("__fe2o3_ir_float_v1_bf16_to_f32")
                    && arguments.len() == 1
        ));
        assert_eq!(
            to_f32_binding.value().unwrap().1,
            Type::Scalar(ScalarType::F32)
        );
    }

    #[derive(Clone, Copy)]
    enum InfallibleBoundsCallUnwindV1 {
        Unreachable,
        Continue,
        CleanupToAssert,
    }

    #[derive(Clone, Copy)]
    enum InfallibleBoundsLengthGuardV1 {
        DirectSwitch,
        EqualTrue,
        NotEqualFalse,
    }

    #[derive(Clone, Copy)]
    enum InfallibleBoundsLengthSourceV1 {
        Length,
        SliceReferenceMetadata,
        RawSliceMetadata,
    }

    #[derive(Clone, Copy)]
    struct InfallibleBoundsFixtureOptionsV1 {
        workgroup_x: u32,
        guarded_length: u128,
        bypass_guard: bool,
        fake_length: bool,
        redefine_index: bool,
        substitute_condition_length: bool,
        call_unwind: InfallibleBoundsCallUnwindV1,
        length_guard: InfallibleBoundsLengthGuardV1,
        alias_length: bool,
        length_source: InfallibleBoundsLengthSourceV1,
    }

    impl Default for InfallibleBoundsFixtureOptionsV1 {
        fn default() -> Self {
            Self {
                workgroup_x: 64,
                guarded_length: 64,
                bypass_guard: false,
                fake_length: false,
                redefine_index: false,
                substitute_condition_length: false,
                call_unwind: InfallibleBoundsCallUnwindV1::Unreachable,
                length_guard: InfallibleBoundsLengthGuardV1::DirectSwitch,
                alias_length: false,
                length_source: InfallibleBoundsLengthSourceV1::Length,
            }
        }
    }

    fn infallible_bounds_fixture_v1(
        options: InfallibleBoundsFixtureOptionsV1,
    ) -> (
        Vec<SemanticTypeDeclV1>,
        Vec<SemanticCallableDeclV1>,
        SemanticFunctionDeclV1,
    ) {
        let unit = SemanticTypeIdV1::from_index(0);
        let u32_ty = SemanticTypeIdV1::from_index(1);
        let u64_ty = SemanticTypeIdV1::from_index(2);
        let bool_ty = SemanticTypeIdV1::from_index(3);
        let source = SemanticSourceProvenanceV1::unavailable();
        let place = |local, ty| {
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap()
        };
        let operand = |local, ty| SemanticOperandV1::Copy(place(local, ty));
        let constant = |ty, value: u128, bytes| {
            SemanticOperandV1::Constant(SemanticConstantV1::new(
                ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, bytes).unwrap()),
            ))
        };
        let assign = |local, ty, kind| {
            SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                    place(local, ty),
                    SemanticRvalueV1::new(ty, kind),
                )),
            )
        };
        let edge = |role, target| {
            SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target))
        };
        let block = |tag, statements, terminator| {
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([tag; 32]),
                source,
                statements,
                SemanticTerminatorV1::new(source, terminator),
            )
            .unwrap()
        };

        let intrinsic_abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([37; 32]),
            SemanticLayoutIdentityV1::from_sha256([38; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(
                u32_ty,
                SemanticAbiPassModeV1::Direct(
                    fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                ),
            ),
        )
        .unwrap();
        let callable = SemanticCallableDeclV1::CompilerIntrinsic {
            binding: SemanticNonBodyCallableBindingV1::new(
                SemanticFunctionIdentityV1::from_sha256([39; 32]),
                SemanticItemDefinitionIdentityV1::from_sha256([40; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([41; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([42; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([43; 32]),
                source,
                intrinsic_abi,
            ),
            operation: SemanticCompilerIntrinsicOperationV1::ThreadIndex(SemanticAxisV1::X),
            operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([44; 32]),
        };

        let call_unwind = match options.call_unwind {
            InfallibleBoundsCallUnwindV1::Unreachable => SemanticUnwindActionV1::Unreachable,
            InfallibleBoundsCallUnwindV1::Continue => SemanticUnwindActionV1::Continue,
            InfallibleBoundsCallUnwindV1::CleanupToAssert => {
                SemanticUnwindActionV1::Cleanup(edge(SemanticEdgeRoleV1::CallUnwind, 2))
            }
        };
        let entry = block(
            45,
            vec![],
            SemanticTerminatorKindV1::Call(
                SemanticDirectCallV1::new_callable(
                    SemanticCallableIdV1::from_index(0),
                    vec![],
                    Some(SemanticCallDestinationV1::new(
                        place(1, u32_ty),
                        edge(SemanticEdgeRoleV1::CallReturn, 1),
                    )),
                    call_unwind,
                )
                .unwrap(),
            ),
        );
        let length_value = if options.fake_length {
            SemanticRvalueKindV1::Use(constant(u64_ty, options.guarded_length, 8))
        } else {
            match options.length_source {
                InfallibleBoundsLengthSourceV1::Length => {
                    SemanticRvalueKindV1::Length(place(5, u64_ty))
                }
                InfallibleBoundsLengthSourceV1::SliceReferenceMetadata => {
                    SemanticRvalueKindV1::Unary {
                        operation: SemanticUnaryOpV1::PointerMetadata,
                        operand: operand(8, SemanticTypeIdV1::from_index(5)),
                    }
                }
                InfallibleBoundsLengthSourceV1::RawSliceMetadata => SemanticRvalueKindV1::Unary {
                    operation: SemanticUnaryOpV1::PointerMetadata,
                    operand: operand(9, SemanticTypeIdV1::from_index(6)),
                },
            }
        };
        let length_local = if options.alias_length { 7 } else { 2 };
        let mut guard_statements = vec![assign(2, u64_ty, length_value)];
        if options.alias_length {
            guard_statements.push(assign(
                length_local,
                u64_ty,
                SemanticRvalueKindV1::Use(operand(2, u64_ty)),
            ));
        }
        let (guard_discriminant, accepted_value) = match options.length_guard {
            InfallibleBoundsLengthGuardV1::DirectSwitch => {
                (operand(length_local, u64_ty), options.guarded_length)
            }
            InfallibleBoundsLengthGuardV1::EqualTrue => {
                guard_statements.push(assign(
                    6,
                    bool_ty,
                    SemanticRvalueKindV1::Binary {
                        operation: SemanticBinaryOpV1::Equal,
                        left: operand(length_local, u64_ty),
                        right: constant(u64_ty, options.guarded_length, 8),
                    },
                ));
                (operand(6, bool_ty), 1)
            }
            InfallibleBoundsLengthGuardV1::NotEqualFalse => {
                guard_statements.push(assign(
                    6,
                    bool_ty,
                    SemanticRvalueKindV1::Binary {
                        operation: SemanticBinaryOpV1::NotEqual,
                        left: operand(length_local, u64_ty),
                        right: constant(u64_ty, options.guarded_length, 8),
                    },
                ));
                (operand(6, bool_ty), 0)
            }
        };
        let guard = block(
            46,
            guard_statements,
            SemanticTerminatorKindV1::SwitchInt {
                discriminant: guard_discriminant,
                targets: SemanticSwitchTargetsV1::new(
                    vec![SemanticSwitchTargetV1::new(
                        accepted_value,
                        edge(SemanticEdgeRoleV1::SwitchValue, 2),
                    )],
                    edge(SemanticEdgeRoleV1::SwitchOtherwise, 4),
                )
                .unwrap(),
            },
        );
        let mut assert_statements = vec![assign(
            3,
            u64_ty,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Integer,
                operand: operand(1, u32_ty),
            },
        )];
        if options.redefine_index {
            assert_statements.push(assign(
                3,
                u64_ty,
                SemanticRvalueKindV1::Use(constant(u64_ty, 0, 8)),
            ));
        }
        let condition_length = if options.substitute_condition_length {
            constant(u64_ty, options.guarded_length, 8)
        } else {
            operand(length_local, u64_ty)
        };
        assert_statements.push(assign(
            4,
            bool_ty,
            SemanticRvalueKindV1::Binary {
                operation: SemanticBinaryOpV1::LessThan,
                left: operand(3, u64_ty),
                right: condition_length,
            },
        ));
        let bounds_assert = block(
            47,
            assert_statements,
            SemanticTerminatorKindV1::Assert {
                condition: operand(4, bool_ty),
                expected: true,
                message: SemanticAssertMessageV1::BoundsCheck {
                    length: operand(length_local, u64_ty),
                    index: operand(3, u64_ty),
                },
                target: edge(SemanticEdgeRoleV1::AssertSuccess, 3),
                unwind: SemanticUnwindActionV1::Unreachable,
            },
        );
        let exit = block(48, vec![], SemanticTerminatorKindV1::Return);
        let rejected = block(
            49,
            vec![],
            if options.bypass_guard {
                SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 2))
            } else {
                SemanticTerminatorKindV1::Return
            },
        );
        let function_abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([50; 32]),
            SemanticLayoutIdentityV1::from_sha256([51; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            3,
            vec![
                SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
                    u64_ty,
                    SemanticAbiPassModeV1::Direct(
                        fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                    ),
                )),
                SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
                    SemanticTypeIdV1::from_index(5),
                    SemanticAbiPassModeV1::Pair {
                        first:
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                        second:
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                    },
                )),
                SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
                    SemanticTypeIdV1::from_index(6),
                    SemanticAbiPassModeV1::Pair {
                        first:
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                        second:
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
                    },
                )),
            ],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let locals = [
            (unit, SemanticLocalRoleV1::Return),
            (u32_ty, SemanticLocalRoleV1::Temporary),
            (u64_ty, SemanticLocalRoleV1::Temporary),
            (u64_ty, SemanticLocalRoleV1::Temporary),
            (bool_ty, SemanticLocalRoleV1::Temporary),
            (u64_ty, SemanticLocalRoleV1::Argument(0)),
            (bool_ty, SemanticLocalRoleV1::Temporary),
            (u64_ty, SemanticLocalRoleV1::Temporary),
            (
                SemanticTypeIdV1::from_index(5),
                SemanticLocalRoleV1::Argument(1),
            ),
            (
                SemanticTypeIdV1::from_index(6),
                SemanticLocalRoleV1::Argument(2),
            ),
        ]
        .into_iter()
        .enumerate()
        .map(|(local, (ty, role))| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([52 + local as u8; 32]),
                ty,
                role,
                source,
            )
        })
        .collect();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([70; 32]),
            SemanticFunctionRoleV1::KernelRoot,
            SemanticItemDefinitionIdentityV1::from_sha256([61; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([62; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([63; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([64; 32]),
            source,
            function_abi,
            locals,
            SemanticBlockIdV1::from_index(0),
            vec![entry, guard, bounds_assert, exit, rejected],
        )
        .unwrap();
        (
            vec![
                unit_type(),
                unsigned_scalar_type(33, 32),
                u64_type(),
                bool_type(),
                SemanticTypeDeclV1::new(
                    SemanticTypeIdentityV1::from_sha256([65; 32]),
                    SemanticLayoutIdentityV1::from_sha256([66; 32]),
                    SemanticTypeLayoutV1::new(None, 4).unwrap(),
                    SemanticTypeShapeV1::Slice { element: u32_ty },
                ),
                SemanticTypeDeclV1::new(
                    SemanticTypeIdentityV1::from_sha256([67; 32]),
                    SemanticLayoutIdentityV1::from_sha256([68; 32]),
                    SemanticTypeLayoutV1::new(Some(16), 8).unwrap(),
                    SemanticTypeShapeV1::Pointer(
                        fe2o3_mir_model::semantic_mir_v1::SemanticPointerTypeV1::new_with_kind(
                            SemanticTypeIdV1::from_index(4),
                            SemanticPointerKindV1::Reference,
                            SemanticMutabilityV1::Immutable,
                            1,
                            64,
                            SemanticPointerMetadataV1::SliceLength,
                        )
                        .unwrap(),
                    ),
                ),
                SemanticTypeDeclV1::new(
                    SemanticTypeIdentityV1::from_sha256([69; 32]),
                    SemanticLayoutIdentityV1::from_sha256([70; 32]),
                    SemanticTypeLayoutV1::new(Some(16), 8).unwrap(),
                    SemanticTypeShapeV1::Pointer(
                        fe2o3_mir_model::semantic_mir_v1::SemanticPointerTypeV1::new_with_kind(
                            SemanticTypeIdV1::from_index(4),
                            SemanticPointerKindV1::Raw,
                            SemanticMutabilityV1::Immutable,
                            1,
                            64,
                            SemanticPointerMetadataV1::SliceLength,
                        )
                        .unwrap(),
                    ),
                ),
            ],
            vec![callable],
            function,
        )
    }

    #[test]
    fn exact_launch_and_dominating_slice_length_prove_only_the_matching_bounds_assert() {
        for call_unwind in [
            InfallibleBoundsCallUnwindV1::Unreachable,
            InfallibleBoundsCallUnwindV1::Continue,
        ] {
            for length_guard in [
                InfallibleBoundsLengthGuardV1::DirectSwitch,
                InfallibleBoundsLengthGuardV1::EqualTrue,
                InfallibleBoundsLengthGuardV1::NotEqualFalse,
            ] {
                for alias_length in [false, true] {
                    for length_source in [
                        InfallibleBoundsLengthSourceV1::Length,
                        InfallibleBoundsLengthSourceV1::SliceReferenceMetadata,
                    ] {
                        let options = InfallibleBoundsFixtureOptionsV1 {
                            call_unwind,
                            length_guard,
                            alias_length,
                            length_source,
                            ..Default::default()
                        };
                        let (types, callables, function) = infallible_bounds_fixture_v1(options);
                        assert_eq!(
                            InfallibleBoundsAssertAnalysisV1::analyze(
                                &types,
                                &callables,
                                &function,
                                [64, 1, 1],
                            )
                            .unwrap(),
                            BTreeSet::from([2]),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn infallible_bounds_proof_rejects_substitutions_and_bypass_paths() {
        for options in [
            InfallibleBoundsFixtureOptionsV1 {
                workgroup_x: 128,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                guarded_length: 32,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                bypass_guard: true,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                fake_length: true,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                redefine_index: true,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                substitute_condition_length: true,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                call_unwind: InfallibleBoundsCallUnwindV1::CleanupToAssert,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                guarded_length: 32,
                length_guard: InfallibleBoundsLengthGuardV1::EqualTrue,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                fake_length: true,
                length_guard: InfallibleBoundsLengthGuardV1::NotEqualFalse,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                bypass_guard: true,
                length_guard: InfallibleBoundsLengthGuardV1::NotEqualFalse,
                ..Default::default()
            },
            InfallibleBoundsFixtureOptionsV1 {
                length_source: InfallibleBoundsLengthSourceV1::RawSliceMetadata,
                ..Default::default()
            },
        ] {
            let (types, callables, function) = infallible_bounds_fixture_v1(options);
            assert!(
                InfallibleBoundsAssertAnalysisV1::analyze(
                    &types,
                    &callables,
                    &function,
                    [options.workgroup_x, 1, 1],
                )
                .unwrap()
                .is_empty(),
            );
        }
    }

    fn operand_contract(role: SemanticMfmaOperandRoleV1) -> SemanticMfmaOperandContractV1 {
        SemanticMfmaOperandContractV1 {
            role,
            profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
            register_distribution: SemanticMfmaRegisterDistributionV1::Tile16x16,
            wave_width: 64,
        }
    }

    fn accumulator_contract() -> SemanticMfmaAccumulatorContractV1 {
        SemanticMfmaAccumulatorContractV1 {
            profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
            distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
            wave_width: 64,
        }
    }

    #[test]
    fn production_semantic_kir_rejects_the_retired_option_load_and_accepts_v2() {
        let legacy = SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoad {
            option_fragment: SemanticTypeIdV1::from_index(0),
            view: SemanticTypeIdV1::from_index(1),
            lane: SemanticTypeIdV1::from_index(2),
            fragment: SemanticTypeIdV1::from_index(3),
            contract: operand_contract(SemanticMfmaOperandRoleV1::A),
            storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
        };
        assert!(matches!(
            require_current_production_intrinsic_v1(&legacy),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                function: 0,
                block: None,
                statement: None,
                detail: "the retired Option-returning BF16 matrix load is not admitted; use Bf16MatrixLoadZeroFilledV2",
            })
        ));

        let zero_filled = SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoadZeroFilledV2 {
            fragment: SemanticTypeIdV1::from_index(3),
            view: SemanticTypeIdV1::from_index(1),
            lane: SemanticTypeIdV1::from_index(2),
            contract: operand_contract(SemanticMfmaOperandRoleV1::A),
            storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
        };
        assert!(require_current_production_intrinsic_v1(&zero_filled).is_ok());
    }

    #[test]
    fn mfma_operand_roles_map_independent_nonzero_bases_to_row_major_coordinates() {
        let minor_base = 11_u64;
        let reduction_base = 37_u64;
        let lane_minor = 5_u64;
        let lane_group = 8_u64;
        let component = 3_u64;
        let stride = 101_u64;

        let a = semantic_mfma_operand_bases_v1(
            SemanticMfmaOperandRoleV1::A,
            minor_base,
            reduction_base,
        );
        let b = semantic_mfma_operand_bases_v1(
            SemanticMfmaOperandRoleV1::B,
            reduction_base,
            minor_base,
        );
        assert_eq!(a, b);

        let minor = a.minor + lane_minor;
        let reduction = a.reduction + lane_group + component;
        assert_eq!(minor, 16);
        assert_eq!(reduction, 48);
        assert_eq!(minor * stride + reduction, 1_664);
        assert_eq!(reduction * stride + minor, 4_864);
    }

    #[test]
    fn mfma_operand_role_mapping_exposes_hostile_swapped_contracts() {
        let minor_base = 11_u64;
        let reduction_base = 37_u64;
        let expected = SemanticMfmaOperandBasesV1 {
            minor: minor_base,
            reduction: reduction_base,
        };

        assert_eq!(
            semantic_mfma_operand_bases_v1(
                SemanticMfmaOperandRoleV1::A,
                minor_base,
                reduction_base,
            ),
            expected
        );
        assert_eq!(
            semantic_mfma_operand_bases_v1(
                SemanticMfmaOperandRoleV1::B,
                reduction_base,
                minor_base,
            ),
            expected
        );
        assert_ne!(
            semantic_mfma_operand_bases_v1(
                SemanticMfmaOperandRoleV1::B,
                minor_base,
                reduction_base,
            ),
            expected
        );
        assert_ne!(
            semantic_mfma_operand_bases_v1(
                SemanticMfmaOperandRoleV1::A,
                reduction_base,
                minor_base,
            ),
            expected
        );
    }

    #[test]
    fn promoted_accumulator_preserves_current_wave_without_a_phi_token() {
        let descriptor = SemanticPromotedBindingV1::AccumulatorFragment {
            contract: accumulator_contract(),
        };
        let values = (10..14)
            .map(|id| (ValueId(id), Type::Scalar(ScalarType::F32)))
            .collect::<Vec<_>>();
        let binding = SemanticValueBindingV1::AccumulatorFragment {
            values: values.clone(),
            contract: accumulator_contract(),
            wave: SemanticCurrentWaveV1::new(64),
        };

        let transport = descriptor.transport_values(&binding).unwrap();
        assert_eq!(transport, values);
        let definitions = transport
            .iter()
            .map(|(id, ty)| ValueDef::new(*id, ty.clone()))
            .collect::<Vec<_>>();
        assert!(matches!(
            descriptor
                .binding_from_transport(
                    std::slice::from_ref(&unit_type()),
                    SemanticTypeIdV1::from_index(0),
                    &definitions,
                )
                .unwrap(),
            SemanticValueBindingV1::AccumulatorFragment {
                values: reconstructed,
                contract,
                wave,
            } if reconstructed == values
                && contract == accumulator_contract()
                && wave == SemanticCurrentWaveV1::new(64)
        ));
    }

    #[test]
    fn promoted_matrix_fragment_preserves_layout_contract_and_current_wave() {
        let contract = operand_contract(SemanticMfmaOperandRoleV1::A);
        let descriptor = SemanticPromotedBindingV1::MatrixFragment {
            contract,
            storage_layout: SemanticMfmaStorageLayoutV1::LdsXor4,
        };
        let values = (20..24)
            .map(|id| (ValueId(id), Type::Scalar(ScalarType::Bf16)))
            .collect::<Vec<_>>();
        let binding = SemanticValueBindingV1::MatrixFragment {
            values: values.clone(),
            contract,
            storage_layout: SemanticMfmaStorageLayoutV1::LdsXor4,
            wave: SemanticCurrentWaveV1::new(64),
        };

        let transport = descriptor.transport_values(&binding).unwrap();
        assert_eq!(transport.len(), 4);
        let definitions = transport
            .iter()
            .map(|(id, ty)| ValueDef::new(*id, ty.clone()))
            .collect::<Vec<_>>();
        assert!(matches!(
            descriptor
                .binding_from_transport(
                    std::slice::from_ref(&unit_type()),
                    SemanticTypeIdV1::from_index(0),
                    &definitions,
                )
                .unwrap(),
            SemanticValueBindingV1::MatrixFragment {
                values: reconstructed,
                contract: reconstructed_contract,
                storage_layout: SemanticMfmaStorageLayoutV1::LdsXor4,
                wave,
            } if reconstructed == values
                && reconstructed_contract == contract
                && wave == SemanticCurrentWaveV1::new(64)
        ));
    }

    #[test]
    fn promoted_fragment_rejects_forged_or_conflicting_metadata() {
        let descriptor = SemanticPromotedBindingV1::AccumulatorFragment {
            contract: accumulator_contract(),
        };
        let ordinary = SemanticValueBindingV1::Aggregate(
            (0..4)
                .map(|id| SemanticValueBindingV1::Value {
                    id: ValueId(id),
                    ty: Type::Scalar(ScalarType::F32),
                })
                .collect(),
        );
        assert_eq!(
            descriptor.transport_values(&ordinary).unwrap_err(),
            "promoted accumulator fragment lacks its authenticated producer metadata"
        );

        let mut wrong_contract = accumulator_contract();
        wrong_contract.wave_width = 32;
        let wrong = SemanticValueBindingV1::AccumulatorFragment {
            values: (0..4)
                .map(|id| (ValueId(id), Type::Scalar(ScalarType::F32)))
                .collect(),
            contract: wrong_contract,
            wave: SemanticCurrentWaveV1::new(32),
        };
        assert!(descriptor.transport_values(&wrong).is_err());

        let wrong_wave = SemanticValueBindingV1::AccumulatorFragment {
            values: (0..4)
                .map(|id| (ValueId(id), Type::Scalar(ScalarType::F32)))
                .collect(),
            contract: accumulator_contract(),
            wave: SemanticCurrentWaveV1::new(32),
        };
        assert!(descriptor.transport_values(&wrong_wave).is_err());

        let mut bindings = BTreeMap::new();
        let ty = SemanticTypeIdV1::from_index(7);
        insert_compiler_issued_ssa_binding_v1(&mut bindings, ty, descriptor).unwrap();
        assert!(
            insert_compiler_issued_ssa_binding_v1(
                &mut bindings,
                ty,
                SemanticPromotedBindingV1::AccumulatorFragment {
                    contract: wrong_contract,
                },
            )
            .unwrap_err()
            .to_string()
            .contains("conflicting compiler-issued contracts")
        );
    }

    #[test]
    fn promoted_fragment_rejects_changed_component_count_and_types() {
        let descriptor = SemanticPromotedBindingV1::AccumulatorFragment {
            contract: accumulator_contract(),
        };
        let mut definitions = (0..4)
            .map(|id| ValueDef::new(ValueId(id), Type::Scalar(ScalarType::F32)))
            .collect::<Vec<_>>();
        definitions.push(ValueDef::new(ValueId(4), Type::INDEX));
        assert!(
            descriptor
                .binding_from_transport(
                    std::slice::from_ref(&unit_type()),
                    SemanticTypeIdV1::from_index(0),
                    &definitions,
                )
                .unwrap_err()
                .to_string()
                .contains("component types changed")
        );

        definitions.pop();
        definitions[0] = ValueDef::new(ValueId(0), Type::Scalar(ScalarType::F64));
        assert!(
            descriptor
                .binding_from_transport(
                    std::slice::from_ref(&unit_type()),
                    SemanticTypeIdV1::from_index(0),
                    &definitions,
                )
                .unwrap_err()
                .to_string()
                .contains("component types changed")
        );
    }

    #[test]
    fn current_wave_lane_rejects_plain_values_and_wrong_widths() {
        let block = SemanticBlockIdV1::from_index(3);
        let plain = SemanticValueBindingV1::Value {
            id: ValueId(7),
            ty: Type::Scalar(ScalarType::U32),
        };
        assert!(require_current_wave_lane(block, plain, 64, "lane authority").is_err());

        let wrong_width = SemanticValueBindingV1::WaveLane {
            value: ValueId(8),
            wave: SemanticCurrentWaveV1::new(32),
        };
        assert!(require_current_wave_lane(block, wrong_width, 64, "lane authority").is_err());

        let lane = SemanticValueBindingV1::WaveLane {
            value: ValueId(9),
            wave: SemanticCurrentWaveV1::new(64),
        };
        assert_eq!(
            require_current_wave_lane(block, lane, 64, "lane authority").unwrap(),
            (ValueId(9), SemanticCurrentWaveV1::new(64))
        );
    }

    #[test]
    fn repeated_current_lane_reads_share_physical_wave_provenance() {
        let block = SemanticBlockIdV1::from_index(4);
        let first = SemanticValueBindingV1::WaveLane {
            value: ValueId(20),
            wave: SemanticCurrentWaveV1::new(64),
        };
        let second = SemanticValueBindingV1::WaveLane {
            value: ValueId(21),
            wave: SemanticCurrentWaveV1::new(64),
        };
        let (first_value, first_wave) =
            require_current_wave_lane(block, first, 64, "lane authority").unwrap();
        let (second_value, second_wave) =
            require_current_wave_lane(block, second, 64, "lane authority").unwrap();

        assert_ne!(first_value, second_value);
        assert_eq!(first_wave, second_wave);
    }

    #[test]
    fn ordinary_aggregate_transport_does_not_gain_fragment_authority() {
        let binding = SemanticValueBindingV1::Aggregate(
            (0..4)
                .map(|id| SemanticValueBindingV1::Value {
                    id: ValueId(id),
                    ty: Type::Scalar(ScalarType::F32),
                })
                .collect(),
        );
        let values = SemanticPromotedBindingV1::Ordinary
            .transport_values(&binding)
            .unwrap();
        assert_eq!(values.len(), 4);
        assert!(
            values
                .iter()
                .all(|(_, ty)| *ty == Type::Scalar(ScalarType::F32))
        );
    }

    struct OperationSpanFixture {
        expected: [ExpectedSemanticKirBlockCoverageV1; 1],
        target: Vec<BasicBlock>,
        blocks: [SemanticKirBlockCorrespondenceV1; 1],
        statements: [SemanticKirStatementOperationSpanV1; 2],
        terminators: [SemanticKirTerminatorOperationSpanV1; 1],
    }

    fn operation_span_fixture() -> OperationSpanFixture {
        let semantic_function = SemanticFunctionIdV1::from_index(0);
        let semantic_block = SemanticBlockIdV1::from_index(7);
        let kernel_ir_block = BlockId(7);
        let expected = [ExpectedSemanticKirBlockCoverageV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block,
            source_statement_count: 2,
        }];
        let blocks = [SemanticKirBlockCorrespondenceV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block,
            source_statement_count: 2,
        }];
        let statements = [
            SemanticKirStatementOperationSpanV1 {
                semantic_function,
                semantic_block,
                statement_ordinal: 0,
                kernel_ir_block,
                first_operation_ordinal: 0,
                operation_count: 0,
            },
            SemanticKirStatementOperationSpanV1 {
                semantic_function,
                semantic_block,
                statement_ordinal: 1,
                kernel_ir_block,
                first_operation_ordinal: 0,
                operation_count: 2,
            },
        ];
        let terminators = [SemanticKirTerminatorOperationSpanV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block,
            first_operation_ordinal: 2,
            operation_count: 1,
        }];
        let operation = AmdGpuDiagnosticOperation::Trap.operation(None);
        let mut target = BasicBlock::new(kernel_ir_block);
        target.operations = vec![operation; 3];
        target.terminator = Some(Terminator::Return { values: vec![] });
        OperationSpanFixture {
            expected,
            target: vec![target],
            blocks,
            statements,
            terminators,
        }
    }

    #[test]
    fn zero_sized_arrays_are_bounded_by_structure_not_only_scalar_components() {
        let unit = unit_type();
        let array = SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([3; 32]),
            SemanticLayoutIdentityV1::from_sha256([4; 32]),
            unit.layout().clone(),
            SemanticTypeShapeV1::Array {
                element: SemanticTypeIdV1::from_index(0),
                length: u64::MAX,
            },
        );
        let types = [unit, array];

        let lowering = lower_ssa_value_types(&types, SemanticTypeIdV1::from_index(1))
            .expect_err("huge zero-sized array must fail before iteration");
        assert!(lowering.to_string().contains("array length is too large"));

        let binding = binding_from_value_defs(&types, SemanticTypeIdV1::from_index(1), &[])
            .expect_err("huge zero-sized binding must fail before allocation");
        assert!(binding.to_string().contains("array length is too large"));
    }

    #[test]
    fn operation_spans_cover_zero_multi_and_terminator_emission_exactly_once() {
        let fixture = operation_span_fixture();
        assert!(validate_operation_correspondence_layout(
            &fixture.expected,
            &fixture.target,
            &fixture.blocks,
            &fixture.statements,
            &fixture.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn operation_span_validation_rejects_statement_omission() {
        let fixture = operation_span_fixture();
        assert!(!validate_operation_correspondence_layout(
            &fixture.expected,
            &fixture.target,
            &fixture.blocks,
            &fixture.statements[..1],
            &fixture.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn operation_span_validation_rejects_overlap() {
        let mut fixture = operation_span_fixture();
        fixture.statements[0].operation_count = 1;
        assert!(!validate_operation_correspondence_layout(
            &fixture.expected,
            &fixture.target,
            &fixture.blocks,
            &fixture.statements,
            &fixture.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn operation_span_validation_rejects_terminator_gaps_and_trailing_operations() {
        let mut gap = operation_span_fixture();
        gap.terminators[0].first_operation_ordinal = 1;
        assert!(!validate_operation_correspondence_layout(
            &gap.expected,
            &gap.target,
            &gap.blocks,
            &gap.statements,
            &gap.terminators,
            &[],
            None,
        ));

        let mut trailing = operation_span_fixture();
        trailing.target[0]
            .operations
            .push(AmdGpuDiagnosticOperation::Trap.operation(None));
        assert!(!validate_operation_correspondence_layout(
            &trailing.expected,
            &trailing.target,
            &trailing.blocks,
            &trailing.statements,
            &trailing.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn operation_span_validation_rejects_target_block_substitution() {
        let mut fixture = operation_span_fixture();
        fixture.target[0].id = BlockId(8);
        assert!(!validate_operation_correspondence_layout(
            &fixture.expected,
            &fixture.target,
            &fixture.blocks,
            &fixture.statements,
            &fixture.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn operation_span_validation_rejects_source_substitution() {
        let mut fixture = operation_span_fixture();
        fixture.statements[1].statement_ordinal = 0;
        assert!(!validate_operation_correspondence_layout(
            &fixture.expected,
            &fixture.target,
            &fixture.blocks,
            &fixture.statements,
            &fixture.terminators,
            &[],
            None,
        ));
    }

    #[test]
    fn synthetic_trap_rule_has_exact_block_and_operation_coverage() {
        let semantic_function = SemanticFunctionIdV1::from_index(0);
        let semantic_block = SemanticBlockIdV1::from_index(0);
        let expected = [ExpectedSemanticKirBlockCoverageV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block: BlockId(0),
            source_statement_count: 0,
        }];
        let blocks = [SemanticKirBlockCorrespondenceV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block: BlockId(0),
            source_statement_count: 0,
        }];
        let terminators = [SemanticKirTerminatorOperationSpanV1 {
            semantic_function,
            semantic_block,
            kernel_ir_block: BlockId(0),
            first_operation_ordinal: 0,
            operation_count: 0,
        }];
        let mut source = BasicBlock::new(BlockId(0));
        source.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments: vec![],
        });
        let mut synthetic_block = BasicBlock::new(BlockId(1));
        synthetic_block
            .operations
            .push(AmdGpuDiagnosticOperation::Trap.operation(None));
        synthetic_block.terminator = Some(Terminator::Unreachable);
        let target = [source, synthetic_block];
        let synthetic = [SemanticKirSyntheticOperationSpanV1 {
            rule: SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap,
            kernel_ir_block: BlockId(1),
            first_operation_ordinal: 0,
            operation_count: 1,
        }];

        assert!(validate_operation_correspondence_layout(
            &expected,
            &target,
            &blocks,
            &[],
            &terminators,
            &synthetic,
            Some(SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap),
        ));
        assert!(!validate_operation_correspondence_layout(
            &expected,
            &target,
            &blocks,
            &[],
            &terminators,
            &[],
            Some(SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap),
        ));

        let mut wrong_trap = target.clone();
        wrong_trap[1].operations[0] =
            Operation::new(Vec::new(), OperationKind::Constant(Constant::Bool(false)));
        assert!(!validate_operation_correspondence_layout(
            &expected,
            &wrong_trap,
            &blocks,
            &[],
            &terminators,
            &synthetic,
            Some(SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap),
        ));
    }

    #[test]
    fn semantic_result_projection_selects_only_the_downcast_variant_payload() {
        let ok_view = SemanticValueBindingV1::Value {
            id: ValueId(17),
            ty: Type::INDEX,
        };
        let error = SemanticValueBindingV1::Value {
            id: ValueId(18),
            ty: Type::Scalar(ScalarType::U32),
        };
        let payloads = BTreeMap::from([(0, vec![ok_view]), (1, vec![error])]);

        assert!(matches!(
            project_enum_payload_field(0, &payloads, 0),
            Ok(SemanticValueBindingV1::Value {
                id: ValueId(17),
                ..
            })
        ));
        assert!(matches!(
            project_enum_payload_field(1, &payloads, 0),
            Ok(SemanticValueBindingV1::Value {
                id: ValueId(18),
                ..
            })
        ));
        let unavailable = project_enum_payload_field(2, &payloads, 0).unwrap();
        assert!(matches!(
            unavailable,
            SemanticValueBindingV1::Unmaterialized
        ));
        assert_eq!(
            unavailable.value().unwrap_err(),
            "unmaterialized enum payload has no ordinary SSA representation"
        );
    }

    #[test]
    fn unmaterialized_wrong_variant_rejects_at_the_lowerer_observation_boundary() {
        let unit = SemanticTypeIdV1::from_index(0);
        let types = [unit_type()];
        let source = SemanticSourceProvenanceV1::unavailable();
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([3; 32]),
            SemanticLayoutIdentityV1::from_sha256([4; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let block = SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([5; 32]),
            source,
            vec![],
            SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([6; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([7; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([8; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([9; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([10; 32]),
            source,
            abi,
            vec![SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([11; 32]),
                unit,
                SemanticLocalRoleV1::Return,
                source,
            )],
            SemanticBlockIdV1::from_index(0),
            vec![block],
        )
        .unwrap();
        let mut lowering = SemanticFunctionLoweringV1::new(
            &types,
            &[],
            &function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            16,
        )
        .unwrap();
        lowering.locals[0] = Some(SemanticValueBindingV1::Unmaterialized);
        let projected = SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(0),
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), unit).unwrap()],
            unit,
        )
        .unwrap();

        assert!(matches!(
            lowering.resolve_place(SemanticBlockIdV1::from_index(0), Some(0), &projected),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "unmaterialized enum payload cannot be observed",
                ..
            })
        ));
    }

    struct GuardedAddressFixture {
        module: Module,
        locations: Vec<FunctionOperationLocation>,
    }

    fn generated_matrix_tail_fixture(component_count: usize) -> GuardedAddressFixture {
        fn emit(
            block: &mut BasicBlock,
            next_value: &mut u32,
            ty: Type,
            kind: OperationKind,
        ) -> ValueId {
            let id = ValueId(*next_value);
            *next_value += 1;
            block
                .operations
                .push(Operation::effect_free(ValueDef::new(id, ty), kind));
            id
        }

        let slice_type = Type::slice(
            Type::Scalar(ScalarType::U16),
            AddressSpace::Global,
            AccessMode::ReadOnly,
        );
        let pointer_type = Type::pointer(
            Type::Scalar(ScalarType::U16),
            AddressSpace::Global,
            AccessMode::ReadOnly,
        );
        let protected_slice = ValueId(0);
        let other_slice = ValueId(1);
        let base_index = ValueId(2);
        let mut next_value = 3;
        let mut block = BasicBlock::new(BlockId(0));
        let data = emit(
            &mut block,
            &mut next_value,
            pointer_type.clone(),
            OperationKind::SliceData {
                slice: protected_slice,
            },
        );
        let length = emit(
            &mut block,
            &mut next_value,
            Type::INDEX,
            OperationKind::SliceLength {
                slice: protected_slice,
            },
        );
        let zero = emit(
            &mut block,
            &mut next_value,
            Type::INDEX,
            OperationKind::Constant(Constant::Index(0)),
        );
        let valid = emit(
            &mut block,
            &mut next_value,
            Type::BOOL,
            OperationKind::Constant(Constant::Bool(true)),
        );
        let fallback = emit(
            &mut block,
            &mut next_value,
            Type::Scalar(ScalarType::U16),
            OperationKind::Constant(Constant::U16(0)),
        );

        let mut locations = Vec::new();
        for component in 0..component_count {
            let component = emit(
                &mut block,
                &mut next_value,
                Type::INDEX,
                OperationKind::Constant(Constant::Index(component as u64)),
            );
            let index = emit(
                &mut block,
                &mut next_value,
                Type::INDEX,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: base_index,
                    rhs: component,
                },
            );
            let in_bounds = emit(
                &mut block,
                &mut next_value,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThan,
                    lhs: index,
                    rhs: length,
                },
            );
            let guard = emit(
                &mut block,
                &mut next_value,
                Type::BOOL,
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: valid,
                    rhs: in_bounds,
                },
            );
            let safe_index = emit(
                &mut block,
                &mut next_value,
                Type::INDEX,
                OperationKind::Select {
                    condition: guard,
                    true_value: index,
                    false_value: zero,
                },
            );
            let pointer = emit(
                &mut block,
                &mut next_value,
                pointer_type.clone(),
                OperationKind::GetElementPointer {
                    base: data,
                    offset: safe_index,
                },
            );
            locations.push(FunctionOperationLocation::new(
                block.id,
                block.operations.len(),
            ));
            emit(
                &mut block,
                &mut next_value,
                Type::Scalar(ScalarType::U16),
                OperationKind::GuardedLoad {
                    pointer,
                    predicate: guard,
                    fallback,
                    access: MemoryAccess::new(AddressSpace::Global, 2),
                },
            );
        }
        block.terminator = Some(Terminator::Return { values: vec![] });

        let mut module = Module::new("generated-matrix-tail");
        module.functions.push(Function::kernel_entry(
            "generated_matrix_tail",
            Signature::new(vec![slice_type.clone(), slice_type, Type::INDEX], vec![]),
            vec![protected_slice, other_slice, base_index],
            vec![block],
        ));
        module.kernels.push(Kernel::new(
            "generated-matrix-tail",
            "generated_matrix_tail",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(64),
            },
        ));
        verify_module(&module).expect("generated guarded-tail fixture must be valid Kernel IR");
        GuardedAddressFixture { module, locations }
    }

    fn guarded_fixture_operations_mut(fixture: &mut GuardedAddressFixture) -> &mut Vec<Operation> {
        &mut fixture.module.functions[0]
            .body
            .as_mut()
            .expect("fixture function is defined")
            .blocks[0]
            .operations
    }

    #[test]
    fn generated_matrix_tail_guarded_loads_have_structural_address_proofs() {
        let fixture = generated_matrix_tail_fixture(4);
        let operation_count = fixture.module.functions[0].body.as_ref().unwrap().blocks[0]
            .operations
            .len();
        assert!(guarded_accesses_have_structural_bounds(
            &fixture.module,
            &fixture.locations,
            operation_count,
        ));
    }

    #[test]
    fn guarded_address_proof_audits_unreported_structural_loads() {
        let mut fixture = generated_matrix_tail_fixture(2);
        let operations = guarded_fixture_operations_mut(&mut fixture);
        let always_true = operations[3].results[0].id;
        let OperationKind::Select { condition, .. } = &mut operations[16].kind else {
            panic!("second fixture select changed");
        };
        *condition = always_true;
        let OperationKind::GuardedLoad { predicate, .. } = &mut operations[18].kind else {
            panic!("second fixture guarded load changed");
        };
        *predicate = always_true;
        verify_module(&fixture.module)
            .expect("hostile unreported predicate remains valid Kernel IR");
        assert!(!guarded_accesses_have_structural_bounds(
            &fixture.module,
            &fixture.locations,
            19,
        ));
    }

    #[test]
    fn guarded_address_proof_rejects_true_predicate_with_unsafe_index() {
        let mut fixture = generated_matrix_tail_fixture(1);
        let operations = guarded_fixture_operations_mut(&mut fixture);
        let always_true = operations[3].results[0].id;
        let OperationKind::Select { condition, .. } = &mut operations[9].kind else {
            panic!("fixture select changed");
        };
        *condition = always_true;
        let OperationKind::GuardedLoad { predicate, .. } = &mut operations[11].kind else {
            panic!("fixture guarded load changed");
        };
        *predicate = always_true;
        verify_module(&fixture.module).expect("hostile true predicate remains structurally valid");
        assert!(!guarded_accesses_have_structural_bounds(
            &fixture.module,
            &fixture.locations,
            12,
        ));
    }

    #[test]
    fn guarded_address_proof_rejects_wrong_bound_slice_and_select() {
        let mut wrong_bound = generated_matrix_tail_fixture(1);
        let OperationKind::Compare { predicate, .. } =
            &mut guarded_fixture_operations_mut(&mut wrong_bound)[7].kind
        else {
            panic!("fixture comparison changed");
        };
        *predicate = ComparePredicate::Equal;
        verify_module(&wrong_bound.module).expect("hostile comparison remains valid Kernel IR");
        let failure = guarded_accesses_have_structural_bounds_result(
            &wrong_bound.module,
            &wrong_bound.locations,
            12,
        )
        .expect_err("wrong comparison must not prove a slice bound");
        assert!(matches!(
            failure,
            ProductionMemoryDischargeFailureV1::GuardedBound {
                location,
                index: ValueId(9),
                slice: ValueId(0),
                ..
            } if location == wrong_bound.locations[0]
        ));
        assert!(
            failure
                .to_string()
                .contains("must be below the length of slice")
        );

        let mut wrong_slice = generated_matrix_tail_fixture(1);
        let OperationKind::SliceLength { slice } =
            &mut guarded_fixture_operations_mut(&mut wrong_slice)[1].kind
        else {
            panic!("fixture slice length changed");
        };
        *slice = ValueId(1);
        verify_module(&wrong_slice.module).expect("other slice has the same valid type");
        assert!(!guarded_accesses_have_structural_bounds(
            &wrong_slice.module,
            &wrong_slice.locations,
            12,
        ));

        let mut wrong_select = generated_matrix_tail_fixture(1);
        let OperationKind::Select { false_value, .. } =
            &mut guarded_fixture_operations_mut(&mut wrong_select)[9].kind
        else {
            panic!("fixture select changed");
        };
        *false_value = ValueId(2);
        verify_module(&wrong_select.module).expect("hostile fallback index remains well typed");
        assert!(!guarded_accesses_have_structural_bounds(
            &wrong_select.module,
            &wrong_select.locations,
            12,
        ));
    }

    #[test]
    fn guarded_address_proof_fails_closed_for_locations_defs_cycles_and_budget() {
        let fixture = generated_matrix_tail_fixture(1);
        assert!(!guarded_accesses_have_structural_bounds(
            &fixture.module,
            &[],
            12,
        ));
        assert!(!guarded_accesses_have_structural_bounds(
            &fixture.module,
            &[fixture.locations[0], fixture.locations[0]],
            12,
        ));
        assert!(!guarded_accesses_have_structural_bounds(
            &fixture.module,
            &fixture.locations,
            0,
        ));

        let mut missing = generated_matrix_tail_fixture(1);
        guarded_fixture_operations_mut(&mut missing).remove(7);
        assert!(!guarded_accesses_have_structural_bounds(
            &missing.module,
            &[FunctionOperationLocation::new(BlockId(0), 10)],
            11,
        ));

        let mut ambiguous = generated_matrix_tail_fixture(1);
        let duplicate = guarded_fixture_operations_mut(&mut ambiguous)[7].clone();
        guarded_fixture_operations_mut(&mut ambiguous).push(duplicate);
        assert!(!guarded_accesses_have_structural_bounds(
            &ambiguous.module,
            &ambiguous.locations,
            13,
        ));

        let mut cyclic = generated_matrix_tail_fixture(1);
        let guard = guarded_fixture_operations_mut(&mut cyclic)[8].results[0].id;
        let OperationKind::Binary { lhs, .. } =
            &mut guarded_fixture_operations_mut(&mut cyclic)[8].kind
        else {
            panic!("fixture guard changed");
        };
        *lhs = guard;
        assert!(!guarded_accesses_have_structural_bounds(
            &cyclic.module,
            &cyclic.locations,
            12,
        ));
    }

    struct UnsupportedIndexCorrelationFixtureV1 {
        module: Module,
        correspondence: SemanticKirCorrespondenceV1,
        reasons: Vec<FormalMemoryIncompleteReason>,
        source: ProductionRankedAccessSourceV1,
    }

    fn unsupported_index_correlation_fixture() -> UnsupportedIndexCorrelationFixtureV1 {
        let slice = Type::slice(
            Type::Scalar(ScalarType::F32),
            AddressSpace::Global,
            AccessMode::ReadOnly,
        );
        let pointer = Type::pointer(
            Type::Scalar(ScalarType::F32),
            AddressSpace::Global,
            AccessMode::ReadOnly,
        );
        let mut block = BasicBlock::new(BlockId(0));
        block.operations = vec![
            Operation::effect_free(
                ValueDef::new(ValueId(1), Type::INDEX),
                OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(2), Type::INDEX),
                OperationKind::Binary {
                    op: BinaryOp::Multiply,
                    lhs: ValueId(1),
                    rhs: ValueId(1),
                },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(3), pointer.clone()),
                OperationKind::SliceData { slice: ValueId(0) },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(4), pointer),
                OperationKind::GetElementPointer {
                    base: ValueId(3),
                    offset: ValueId(2),
                },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(5), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ),
        ];
        block.terminator = Some(Terminator::Return { values: vec![] });
        let mut module = Module::new("unsupported-index-correlation");
        module.functions.push(Function::kernel_entry(
            "unsupported_index_correlation",
            Signature::new(vec![slice], vec![]),
            vec![ValueId(0)],
            vec![block],
        ));
        module.kernels.push(Kernel::new(
            "unsupported-index-correlation",
            "unsupported_index_correlation",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(64),
            },
        ));
        verify_module(&module).expect("unsupported-index fixture must be valid Kernel IR");
        let analysis = fe2o3_kernel_ir::derive_kernel_memory_obligations_for_launch(
            &module,
            &module.kernels[0].id,
            fe2o3_kernel_ir::ExplicitLaunchExtent::Exact {
                rank: 1,
                extents: [64, 1, 1],
            },
            fe2o3_kernel_ir::FormalIndexWidth::Bits64,
        )
        .expect("formal analysis must run");
        let fe2o3_kernel_ir::FormalMemoryObligationAnalysis::Incomplete { reasons, .. } = analysis
        else {
            panic!("nonlinear fixture must remain formally incomplete");
        };
        assert!(matches!(
            reasons.as_slice(),
            [FormalMemoryIncompleteReason::UnsupportedIndexExpression {
                location: FunctionOperationLocation {
                    block: BlockId(0),
                    operation_index: 3,
                },
                index: ValueId(2),
                allocation,
            }] if allocation.parameter_index() == 0
        ));
        let correspondence = SemanticKirCorrespondenceV1 {
            semantic_sha256: [7; 32],
            function_count: 1,
            blocks: vec![SemanticKirBlockCorrespondenceV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                source_statement_count: 1,
            }]
            .into_boxed_slice(),
            statement_operation_spans: vec![SemanticKirStatementOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                statement_ordinal: 0,
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 0,
                operation_count: 5,
            }]
            .into_boxed_slice(),
            terminator_operation_spans: vec![SemanticKirTerminatorOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 5,
                operation_count: 0,
            }]
            .into_boxed_slice(),
            synthetic_operation_spans: Box::new([]),
            parameter_bindings: Box::new([]),
        };
        UnsupportedIndexCorrelationFixtureV1 {
            module,
            correspondence,
            reasons,
            source: ProductionRankedAccessSourceV1::new(0, Some(0), 0, 0, 3),
        }
    }

    fn ranked_correlation_input(
        access: AccessKindAttr,
        allocation_origin: u64,
    ) -> ProductionRankedKernelLoweringInputV1 {
        ranked_correlation_input_for_accesses(&[access], allocation_origin)
    }

    fn ranked_correlation_input_for_accesses(
        accesses: &[AccessKindAttr],
        allocation_origin: u64,
    ) -> ProductionRankedKernelLoweringInputV1 {
        let effects = accesses
            .iter()
            .copied()
            .map(|kind| ProductionRankedOperationV1::Access {
                kind,
                view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0)),
                indices: vec![ProductionRankedValueV1::Local(
                    ProductionRankedValueIdV1::new(1),
                )],
            })
            .collect::<Vec<_>>();
        ranked_correlation_input_for_effects(effects, allocation_origin)
    }

    fn ranked_correlation_input_for_effects(
        effects: Vec<ProductionRankedOperationV1>,
        allocation_origin: u64,
    ) -> ProductionRankedKernelLoweringInputV1 {
        let view = ProductionRankedValueIdV1::new(0);
        let index = ProductionRankedValueIdV1::new(1);
        let has_atomic = effects.iter().any(|operation| {
            matches!(
                operation,
                ProductionRankedOperationV1::AtomicAccess { .. }
                    | ProductionRankedOperationV1::AtomicValueAccess { .. }
            )
        });
        let writable = effects.iter().any(|operation| {
            matches!(
                operation,
                ProductionRankedOperationV1::Access { kind, .. }
                    | ProductionRankedOperationV1::ValueAccess { kind, .. }
                    | ProductionRankedOperationV1::AtomicAccess { kind, .. }
                    | ProductionRankedOperationV1::AtomicValueAccess { kind, .. }
                    if kind.writes_memory()
            )
        });
        let mut operations = vec![
            ProductionRankedOperationV1::ExecutionLayout {
                grid_identity: 1,
                global_extents: [64, 1, 1],
                workgroup_extents: [64, 1, 1],
                subgroup_size: 64,
                full_physical_workgroups: true,
            },
            ProductionRankedOperationV1::ViewInSpace {
                result: view,
                element_width: 32,
                writable,
                shape: vec![64],
                dynamic_extents: vec![],
                memory_space: dialect_kernel::MemorySpaceAttr::Global,
                allocation_origin,
                noalias_class: allocation_origin,
            },
            ProductionRankedOperationV1::InvocationIndex {
                result: index,
                dimension: 0,
                launch_extent: 64,
            },
        ];
        operations.extend(effects);
        let kernel = ProductionRankedKernelV1::new(
            "unsupported_index_correlation",
            0,
            vec![ProductionRankedBlockV1::new(
                operations,
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .expect("ranked correlation fixture must be valid");
        let construction =
            ProductionConstructionV1::ranked_kernel("unsupported_index_module", kernel).unwrap();
        if has_atomic {
            compile_ranked_kernel_for_gfx942_lowering_v1(
                construction,
                ProductionSessionLimitsV1::default(),
                [],
            )
        } else {
            compile_ranked_kernel_for_lowering_v1(
                construction,
                ProductionSessionLimitsV1::default(),
            )
        }
        .expect("ranked correlation fixture must pass mandatory checks")
    }

    #[test]
    fn unsupported_index_discharge_requires_exact_location_index_and_consumer_receipts() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        assert!(unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &fixture.reasons,
            16,
        ));

        let mut wrong_location = fixture.reasons.clone();
        let FormalMemoryIncompleteReason::UnsupportedIndexExpression { location, .. } =
            &mut wrong_location[0]
        else {
            unreachable!();
        };
        *location = FunctionOperationLocation::new(BlockId(0), 2);
        assert!(!unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &wrong_location,
            16,
        ));

        let mut wrong_index = fixture.reasons.clone();
        let FormalMemoryIncompleteReason::UnsupportedIndexExpression { index, .. } =
            &mut wrong_index[0]
        else {
            unreachable!();
        };
        *index = ValueId(1);
        assert!(!unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &wrong_index,
            16,
        ));

        let mut mistranslated = fixture.module.clone();
        let OperationKind::GetElementPointer { offset, .. } =
            &mut mistranslated.functions[0].body.as_mut().unwrap().blocks[0].operations[3].kind
        else {
            unreachable!();
        };
        *offset = ValueId(1);
        assert!(!unsupported_indices_match_ranked_sources(
            &mistranslated,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &fixture.reasons,
            16,
        ));

        let mut ambiguous = fixture.module.clone();
        ambiguous.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .push(Operation::effect_free(
                ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ));
        assert!(!unsupported_indices_match_ranked_sources(
            &ambiguous,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &fixture.reasons,
            16,
        ));
    }

    #[test]
    fn unsupported_index_discharge_correlates_every_consumer_of_one_address() {
        let fixture = unsupported_index_correlation_fixture();
        let mut module = fixture.module.clone();
        module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .push(Operation::effect_free(
                ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ));
        verify_module(&module).expect("shared-address fixture remains valid Kernel IR");
        let mut correspondence = fixture.correspondence.clone();
        correspondence.statement_operation_spans[0].operation_count = 6;
        correspondence.terminator_operation_spans[0].first_operation_ordinal = 6;
        let lowering =
            ranked_correlation_input_for_accesses(&[AccessKindAttr::Read, AccessKindAttr::Read], 1);
        let sources = [
            fixture.source,
            ProductionRankedAccessSourceV1::new(0, Some(0), 1, 0, 4),
        ];

        assert!(unsupported_indices_match_ranked_sources(
            &module,
            &correspondence,
            &lowering,
            &sources,
            &fixture.reasons,
            16,
        ));
        assert!(!unsupported_indices_match_ranked_sources(
            &module,
            &correspondence,
            &lowering,
            &sources[..1],
            &fixture.reasons,
            16,
        ));
        let failure = unsupported_indices_match_ranked_sources_result(
            &module,
            &correspondence,
            &lowering,
            &sources[..1],
            &fixture.reasons,
            16,
        )
        .expect_err("the second access has no ranked receipt");
        assert!(matches!(
            failure,
            ProductionMemoryDischargeFailureV1::Access {
                location: FunctionOperationLocation {
                    block: BlockId(0),
                    operation_index: 5,
                },
                ..
            }
        ));
    }

    #[test]
    fn unsupported_index_discharge_rejects_ranked_source_and_allocation_drift() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        for sources in [
            vec![],
            vec![fixture.source, fixture.source],
            vec![ProductionRankedAccessSourceV1::new(0, None, 0, 0, 3)],
            vec![ProductionRankedAccessSourceV1::new(0, Some(0), 1, 0, 3)],
        ] {
            assert!(!unsupported_indices_match_ranked_sources(
                &fixture.module,
                &fixture.correspondence,
                &lowering,
                &sources,
                &fixture.reasons,
                16,
            ));
        }
        let wrong_kind = ranked_correlation_input(AccessKindAttr::Write, 1);
        assert!(!unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &wrong_kind,
            &[fixture.source],
            &fixture.reasons,
            16,
        ));
        let wrong_allocation = ranked_correlation_input(AccessKindAttr::Read, 2);
        assert!(!unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &wrong_allocation,
            &[fixture.source],
            &fixture.reasons,
            16,
        ));
        assert!(!unsupported_indices_match_ranked_sources(
            &fixture.module,
            &fixture.correspondence,
            &lowering,
            &[fixture.source],
            &fixture.reasons,
            4,
        ));
    }

    #[test]
    fn unsupported_index_correlation_is_bounded_at_the_exact_operation_limit() {
        let mut fixture = unsupported_index_correlation_fixture();
        let operations =
            &mut fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations;
        for raw_id in 6..=16 {
            operations.push(Operation::effect_free(
                ValueDef::new(ValueId(raw_id), Type::INDEX),
                OperationKind::Constant(Constant::Index(u64::from(raw_id))),
            ));
        }
        fixture.correspondence.statement_operation_spans[0].operation_count = 16;
        fixture.correspondence.terminator_operation_spans[0].first_operation_ordinal = 16;
        verify_module(&fixture.module).expect("near-limit fixture remains valid Kernel IR");
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);

        for _ in 0..8 {
            assert!(unsupported_indices_match_ranked_sources(
                &fixture.module,
                &fixture.correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                16,
            ));
            assert!(!unsupported_indices_match_ranked_sources(
                &fixture.module,
                &fixture.correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                15,
            ));
        }
    }

    #[test]
    fn unsupported_index_correlation_rejects_fanout_cycles_and_ambiguous_spans() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        let pointer_type = Type::pointer(
            Type::Scalar(ScalarType::F32),
            AddressSpace::Global,
            AccessMode::ReadOnly,
        );

        let mut storage_exhausted = fixture.module.clone();
        let gep =
            &mut storage_exhausted.functions[0].body.as_mut().unwrap().blocks[0].operations[3];
        for raw_id in 6..=400 {
            gep.results
                .push(ValueDef::new(ValueId(raw_id), pointer_type.clone()));
        }

        let mut fanout = fixture.module.clone();
        let fanout_operations =
            &mut fanout.functions[0].body.as_mut().unwrap().blocks[0].operations;
        for offset in 0..24 {
            let pointer = ValueId(10 + offset);
            fanout_operations.push(Operation::effect_free(
                ValueDef::new(pointer, pointer_type.clone()),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: ValueId(4),
                    to: pointer_type.clone(),
                },
            ));
            fanout_operations.push(Operation::effect_free(
                ValueDef::new(ValueId(100 + offset), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer,
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ));
        }

        let mut cyclic = fixture.module.clone();
        let OperationKind::GetElementPointer { base, .. } =
            &mut cyclic.functions[0].body.as_mut().unwrap().blocks[0].operations[3].kind
        else {
            unreachable!();
        };
        *base = ValueId(4);

        let mut ambiguous_correspondence = fixture.correspondence.clone();
        let duplicate = ambiguous_correspondence.statement_operation_spans[0];
        ambiguous_correspondence.statement_operation_spans =
            vec![duplicate, duplicate].into_boxed_slice();

        for _ in 0..8 {
            assert!(!unsupported_indices_match_ranked_sources(
                &storage_exhausted,
                &fixture.correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                5,
            ));
            assert!(!unsupported_indices_match_ranked_sources(
                &fanout,
                &fixture.correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                64,
            ));
            assert!(!unsupported_indices_match_ranked_sources(
                &cyclic,
                &fixture.correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                16,
            ));
            assert!(!unsupported_indices_match_ranked_sources(
                &fixture.module,
                &ambiguous_correspondence,
                &lowering,
                &[fixture.source],
                &fixture.reasons,
                16,
            ));
        }
    }

    fn validate_translation_fixture(
        fixture: &UnsupportedIndexCorrelationFixtureV1,
        lowering: &ProductionRankedKernelLoweringInputV1,
        sources: &[ProductionRankedAccessSourceV1],
        max_operations: usize,
    ) -> Result<ProductionMirPlironTranslationValidationV1, ProductionMirPlironTranslationErrorV1>
    {
        validate_mir_pliron_translation_v1(
            &fixture.module,
            &fixture.correspondence,
            lowering,
            sources,
            max_operations,
        )
    }

    #[test]
    fn mir_pliron_translation_validation_accepts_exact_effect_bijection() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        let report = validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16)
            .expect("the exact independent projections must reconcile");

        assert_eq!(report.semantic_sha256(), &[7; 32]);
        assert_eq!(report.memory_effects(), 1);
        assert_eq!(report.synchronization_effects(), 0);
        assert_eq!(report.tensor_operations(), 0);
        assert_eq!(report.value_expressions(), 0);
        assert_eq!(report.conservative_ranked_effects(), 0);
        assert!(!report.grants_artifact_or_launch_authority());
    }

    #[test]
    fn mir_pliron_translation_validation_accepts_exact_conservative_allocation_effect() {
        let mut fixture = unsupported_index_correlation_fixture();
        fixture.module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .push(Operation::effect_free(
                ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ));
        verify_module(&fixture.module).expect("multi-load summary fixture remains valid");
        fixture.correspondence.statement_operation_spans[0].operation_count = 6;
        fixture.correspondence.terminator_operation_spans[0].first_operation_ordinal = 6;
        let lowering = ranked_correlation_input_for_effects(
            vec![ProductionRankedOperationV1::AllocationEffect {
                kind: AccessKindAttr::Read,
                memory_space: dialect_kernel::MemorySpaceAttr::Global,
                allocation_origin: 1,
                noalias_class: 1,
            }],
            1,
        );
        let report = validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16)
            .expect("the exact conservative allocation effect must reconcile");

        assert_eq!(report.memory_effects(), 2);
        assert_eq!(report.conservative_ranked_effects(), 1);

        let wrong_allocation = ranked_correlation_input_for_effects(
            vec![ProductionRankedOperationV1::AllocationEffect {
                kind: AccessKindAttr::Read,
                memory_space: dialect_kernel::MemorySpaceAttr::Global,
                allocation_origin: 2,
                noalias_class: 2,
            }],
            1,
        );
        assert!(matches!(
            validate_translation_fixture(&fixture, &wrong_allocation, &[fixture.source], 16),
            Err(ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch { .. })
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_missing_and_extra_effects() {
        let fixture = unsupported_index_correlation_fixture();
        let one = ranked_correlation_input(AccessKindAttr::Read, 1);
        assert!(matches!(
            validate_translation_fixture(&fixture, &one, &[], 16),
            Err(ProductionMirPlironTranslationErrorV1::MissingRankedEffect {
                semantic_block: 0,
                semantic_statement: Some(0),
                semantic_access_ordinal: 0,
            })
        ));

        let two =
            ranked_correlation_input_for_accesses(&[AccessKindAttr::Read, AccessKindAttr::Read], 1);
        let sources = [
            fixture.source,
            ProductionRankedAccessSourceV1::new(0, Some(0), 1, 0, 4),
        ];
        assert!(matches!(
            validate_translation_fixture(&fixture, &two, &sources, 16),
            Err(ProductionMirPlironTranslationErrorV1::ExtraRankedEffect {
                ranked_block: 0,
                ranked_operation: 4,
            })
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_kind_allocation_and_site_drift() {
        let fixture = unsupported_index_correlation_fixture();
        let wrong_kind = ranked_correlation_input(AccessKindAttr::Write, 1);
        assert!(matches!(
            validate_translation_fixture(&fixture, &wrong_kind, &[fixture.source], 16),
            Err(ProductionMirPlironTranslationErrorV1::AccessKindMismatch { .. })
        ));

        let wrong_allocation = ranked_correlation_input(AccessKindAttr::Read, 2);
        assert!(matches!(
            validate_translation_fixture(&fixture, &wrong_allocation, &[fixture.source], 16),
            Err(ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch { .. })
        ));

        let wrong_site = ProductionRankedAccessSourceV1::new(0, None, 0, 0, 3);
        assert!(matches!(
            validate_translation_fixture(&fixture, &wrong_allocation, &[wrong_site], 16),
            Err(ProductionMirPlironTranslationErrorV1::MissingRankedEffect {
                semantic_statement: Some(0),
                ..
            })
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_unattributed_global_effects() {
        let mut fixture = unsupported_index_correlation_fixture();
        fixture.correspondence.statement_operation_spans[0].operation_count = 4;
        fixture.correspondence.terminator_operation_spans[0].first_operation_ordinal = 4;
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        assert!(matches!(
            validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16),
            Err(
                ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect {
                    location: FunctionOperationLocation {
                        block: BlockId(0),
                        operation_index: 4,
                    },
                }
            )
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_unrepresented_address_spaces() {
        for address_space in [AddressSpace::Constant, AddressSpace::Generic] {
            let mut fixture = unsupported_index_correlation_fixture();
            let OperationKind::Load { access, .. } =
                &mut fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations[4]
                    .kind
            else {
                unreachable!();
            };
            access.address_space = address_space;
            let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);

            assert!(matches!(
                validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16),
                Err(
                    ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect {
                        location: FunctionOperationLocation {
                            block: BlockId(0),
                            operation_index: 4,
                        },
                    }
                )
            ));
        }
    }

    #[test]
    fn mir_pliron_translation_validation_binds_atomic_ordering_and_scope() {
        let mut fixture = unsupported_index_correlation_fixture();
        let OperationKind::Load { pointer, access } =
            fixture.module.functions[0].body.as_ref().unwrap().blocks[0].operations[4]
                .kind
                .clone()
        else {
            unreachable!();
        };
        fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations[4].kind =
            OperationKind::Atomic(Atomic {
                kind: AtomicKind::Load,
                pointer,
                value: None,
                compare: None,
                access,
                scope: SynchronizationScope::Device,
                ordering: MemoryOrdering::Acquire,
                failure_ordering: None,
            });
        let ranked = |ordering, scope| {
            ranked_correlation_input_for_effects(
                vec![ProductionRankedOperationV1::AtomicAccess {
                    kind: AccessKindAttr::AtomicRead,
                    ordering,
                    scope,
                    view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0)),
                    indices: vec![ProductionRankedValueV1::Local(
                        ProductionRankedValueIdV1::new(1),
                    )],
                }],
                1,
            )
        };

        assert!(
            validate_translation_fixture(
                &fixture,
                &ranked(
                    dialect_kernel::AtomicOrderingAttr::Acquire,
                    dialect_kernel::AtomicScopeAttr::Agent,
                ),
                &[fixture.source],
                16,
            )
            .is_ok()
        );
        for lowering in [
            ranked(
                dialect_kernel::AtomicOrderingAttr::Relaxed,
                dialect_kernel::AtomicScopeAttr::Agent,
            ),
            ranked(
                dialect_kernel::AtomicOrderingAttr::Acquire,
                dialect_kernel::AtomicScopeAttr::Workgroup,
            ),
        ] {
            assert!(matches!(
                validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16),
                Err(ProductionMirPlironTranslationErrorV1::AtomicContractMismatch { .. })
            ));
        }

        {
            let OperationKind::Atomic(atomic) =
                &mut fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations[4]
                    .kind
            else {
                unreachable!();
            };
            atomic.failure_ordering = Some(MemoryOrdering::Relaxed);
        }
        assert!(matches!(
            validate_translation_fixture(
                &fixture,
                &ranked(
                    dialect_kernel::AtomicOrderingAttr::Acquire,
                    dialect_kernel::AtomicScopeAttr::Agent,
                ),
                &[fixture.source],
                16,
            ),
            Err(ProductionMirPlironTranslationErrorV1::AtomicContractMismatch { .. })
        ));
        {
            let OperationKind::Atomic(atomic) =
                &mut fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations[4]
                    .kind
            else {
                unreachable!();
            };
            atomic.failure_ordering = None;
            atomic.scope = SynchronizationScope::Subgroup;
        }
        assert!(matches!(
            validate_translation_fixture(
                &fixture,
                &ranked(
                    dialect_kernel::AtomicOrderingAttr::Acquire,
                    dialect_kernel::AtomicScopeAttr::Agent,
                ),
                &[fixture.source],
                16,
            ),
            Err(ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect { .. })
        ));
    }

    #[test]
    fn translation_memory_classifier_covers_compound_and_matrix_effects() {
        let stage = Operation::new(
            vec![],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Stage {
                    format: Gfx950LdsTransposeFormatV1::Fp8E4M3,
                    storage: ValueId(1),
                    source_slice: ValueId(2),
                    offset: ValueId(3),
                    rows: ValueId(4),
                    columns: ValueId(5),
                    stride: ValueId(6),
                    token_base: ValueId(7),
                    reduction_base: ValueId(8),
                },
            )),
        );
        assert_eq!(
            kir_memory_accesses_v1(&stage),
            vec![
                (
                    ValueId(2),
                    AccessKindAttr::Read,
                    dialect_kernel::MemorySpaceAttr::Global,
                    None,
                ),
                (
                    ValueId(1),
                    AccessKindAttr::Write,
                    dialect_kernel::MemorySpaceAttr::Workgroup,
                    None,
                ),
            ]
        );
        assert_eq!(
            kir_memory_accesses_v1(&Operation::new(
                vec![],
                OperationKind::Matrix(MatrixOperation::lds_load(
                    ValueId(9),
                    fe2o3_kernel_ir::MatrixElement::Bf16,
                )),
            )),
            vec![(
                ValueId(9),
                AccessKindAttr::Read,
                dialect_kernel::MemorySpaceAttr::Workgroup,
                None,
            )]
        );
        assert_eq!(
            kir_memory_accesses_v1(&Operation::new(
                vec![],
                OperationKind::Matrix(MatrixOperation::lds_store(
                    ValueId(10),
                    [ValueId(11), ValueId(12), ValueId(13), ValueId(14)],
                    fe2o3_kernel_ir::MatrixElement::Bf16,
                )),
            )),
            vec![(
                ValueId(10),
                AccessKindAttr::Write,
                dialect_kernel::MemorySpaceAttr::Workgroup,
                None,
            )]
        );

        let element = fe2o3_kernel_ir::MemoryElementType::Scalar(ScalarType::U32);
        let volatile = Operation::new(
            vec![ValueDef::new(ValueId(21), Type::Scalar(ScalarType::U32))],
            OperationKind::MemoryIntrinsic(MemoryIntrinsicOperation::VolatileLoad {
                pointer: ValueId(20),
                element,
                address_space: AddressSpace::Global,
                layout: fe2o3_kernel_ir::MemoryLayout::new(4, 4),
                contract: fe2o3_kernel_ir::VolatileAccessContract::external_mmio_load(),
            }),
        );
        assert!(kir_memory_accesses_v1(&volatile).is_empty());
        let mut block = BasicBlock::new(BlockId(0));
        block.operations.push(volatile);
        block.terminator = Some(Terminator::Return { values: vec![] });
        let body = Function::kernel_entry(
            "unrepresented_volatile",
            Signature::new(vec![], vec![]),
            vec![],
            vec![block],
        )
        .body
        .unwrap();
        let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 64 };
        let index = build_kir_correlation_index(&body, 1, &mut budget).unwrap();
        assert_eq!(
            index.unmodeled_memory_effects,
            vec![FunctionOperationLocation::new(BlockId(0), 0)],
        );
    }

    fn workgroup_translation_fixture(
        operation: Operation,
    ) -> (Module, SemanticKirCorrespondenceV1) {
        let mut block = BasicBlock::new(BlockId(0));
        block.operations.push(operation);
        block.terminator = Some(Terminator::Return { values: vec![] });
        let mut module = Module::new("workgroup-translation");
        module.functions.push(Function::kernel_entry(
            "workgroup_translation",
            Signature::new(vec![gfx950_lds_transpose_pointer_type_v1()], vec![]),
            vec![ValueId(0)],
            vec![block],
        ));
        module.kernels.push(Kernel::new(
            "workgroup-translation",
            "workgroup_translation",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(64),
            },
        ));
        let correspondence = SemanticKirCorrespondenceV1 {
            semantic_sha256: [10; 32],
            function_count: 1,
            blocks: vec![SemanticKirBlockCorrespondenceV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                source_statement_count: 1,
            }]
            .into_boxed_slice(),
            statement_operation_spans: vec![SemanticKirStatementOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                statement_ordinal: 0,
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 0,
                operation_count: 1,
            }]
            .into_boxed_slice(),
            terminator_operation_spans: vec![SemanticKirTerminatorOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 1,
                operation_count: 0,
            }]
            .into_boxed_slice(),
            synthetic_operation_spans: Box::new([]),
            parameter_bindings: Box::new([]),
        };
        (module, correspondence)
    }

    fn ranked_gfx950_transpose_lifecycle(fp8: bool) -> ProductionRankedKernelLoweringInputV1 {
        let (allocation_origin, noalias_class) = if fp8 {
            (
                dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
                dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
            )
        } else {
            (
                dialect_kernel::GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
                dialect_kernel::GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
            )
        };
        let effect = |kind| ProductionRankedOperationV1::AllocationEffect {
            kind,
            memory_space: dialect_kernel::MemorySpaceAttr::Workgroup,
            allocation_origin,
            noalias_class,
        };
        let kernel = ProductionRankedKernelV1::new(
            "workgroup_translation",
            0,
            vec![ProductionRankedBlockV1::new(
                vec![
                    ProductionRankedOperationV1::ExecutionLayout {
                        grid_identity: 10,
                        global_extents: [64, 1, 1],
                        workgroup_extents: [64, 1, 1],
                        subgroup_size: 64,
                        full_physical_workgroups: true,
                    },
                    effect(AccessKindAttr::Write),
                    ProductionRankedOperationV1::Barrier {
                        execution_scope: dialect_gpu::HierarchyAttr::Workgroup,
                        memory_scope: dialect_gpu::MemoryScopeAttr::Workgroup,
                        address_space: dialect_gpu::AddressSpaceAttr::Workgroup,
                        order: dialect_gpu::MemoryOrderAttr::AcquireRelease,
                    },
                    effect(AccessKindAttr::Read),
                ],
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .expect("exact reserved transpose lifecycle");
        compile_ranked_kernel_for_lowering_v1(
            ProductionConstructionV1::ranked_kernel("workgroup_translation", kernel).unwrap(),
            ProductionSessionLimitsV1::default(),
        )
        .expect("exact reserved transpose lifecycle reaches lowering")
    }

    #[test]
    fn mir_pliron_translation_binds_reserved_workgroup_effects_to_format_and_operation() {
        let read_fp8 = Operation::new(
            vec![],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Read {
                    format: Gfx950LdsTransposeFormatV1::Fp8E4M3,
                    storage: ValueId(0),
                },
            )),
        );
        assert_eq!(
            expected_gfx950_workgroup_allocation_identity_v1(&read_fp8, 0),
            Some((
                dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
                dialect_kernel::GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
            ))
        );
        assert_eq!(
            expected_gfx950_workgroup_allocation_identity_v1(&read_fp8, 1),
            None
        );

        let (module, correspondence) = workgroup_translation_fixture(read_fp8);
        let fp4_lowering = ranked_gfx950_transpose_lifecycle(false);
        let read_source = ProductionRankedAccessSourceV1::new(0, Some(0), 0, 0, 3);
        assert!(matches!(
            validate_mir_pliron_translation_v1(
                &module,
                &correspondence,
                &fp4_lowering,
                &[read_source],
                8,
            ),
            Err(ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch { .. })
        ));

        let ordinary_lds = Operation::new(
            vec![],
            OperationKind::Matrix(MatrixOperation::lds_load(
                ValueId(0),
                fe2o3_kernel_ir::MatrixElement::Bf16,
            )),
        );
        assert_eq!(
            expected_gfx950_workgroup_allocation_identity_v1(&ordinary_lds, 0),
            None
        );
        let (module, correspondence) = workgroup_translation_fixture(ordinary_lds);
        let fp8_lowering = ranked_gfx950_transpose_lifecycle(true);
        assert!(matches!(
            validate_mir_pliron_translation_v1(
                &module,
                &correspondence,
                &fp8_lowering,
                &[read_source],
                8,
            ),
            Err(ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch { .. })
        ));
    }

    #[test]
    fn translation_synchronization_classifier_covers_implicit_publish_barriers() {
        let publish = Operation::new(
            vec![],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Publish {
                    format: Gfx950LdsTransposeFormatV1::Fp8E4M3,
                    storage: ValueId(1),
                },
            )),
        );
        let mut block = BasicBlock::new(BlockId(0));
        block.operations.push(publish);
        block.operations.push(Operation::new(
            vec![],
            OperationKind::Fence(fe2o3_kernel_ir::Fence {
                memory_scope: SynchronizationScope::Device,
                semantics: BarrierSemantics::new(MemoryOrdering::Release, [AddressSpace::Global]),
            }),
        ));
        block.terminator = Some(Terminator::Return { values: vec![] });
        let body = Function::kernel_entry(
            "publish_barrier",
            Signature::new(vec![], vec![]),
            vec![],
            vec![block],
        )
        .body
        .unwrap();

        assert_eq!(
            kir_synchronization_contracts_v1(&body).unwrap(),
            vec![
                NormalizedSynchronizationV1 {
                    execution_scope: None,
                    memory_scope: 3,
                    ordering: 2,
                    address_space: 2,
                },
                NormalizedSynchronizationV1 {
                    execution_scope: Some(2),
                    memory_scope: 2,
                    ordering: 3,
                    address_space: 1,
                },
            ]
        );
    }

    #[test]
    fn compound_operation_effects_receive_stable_source_ordinals() {
        let stage = Operation::new(
            vec![],
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1::full(
                Gfx950LdsTransposeOperationKindV1::Stage {
                    format: Gfx950LdsTransposeFormatV1::Fp4E2M1,
                    storage: ValueId(1),
                    source_slice: ValueId(2),
                    offset: ValueId(3),
                    rows: ValueId(4),
                    columns: ValueId(5),
                    stride: ValueId(6),
                    token_base: ValueId(7),
                    reduction_base: ValueId(8),
                },
            )),
        );
        let mut block = BasicBlock::new(BlockId(0));
        block.operations.push(stage);
        block.terminator = Some(Terminator::Return { values: vec![] });
        let function = Function::kernel_entry(
            "compound_effects",
            Signature::new(vec![], vec![]),
            vec![],
            vec![block],
        );
        let body = function.body.as_ref().unwrap();
        let correspondence = SemanticKirCorrespondenceV1 {
            semantic_sha256: [9; 32],
            function_count: 1,
            blocks: vec![SemanticKirBlockCorrespondenceV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                source_statement_count: 1,
            }]
            .into_boxed_slice(),
            statement_operation_spans: vec![SemanticKirStatementOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                statement_ordinal: 0,
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 0,
                operation_count: 1,
            }]
            .into_boxed_slice(),
            terminator_operation_spans: vec![SemanticKirTerminatorOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 1,
                operation_count: 0,
            }]
            .into_boxed_slice(),
            synthetic_operation_spans: Box::new([]),
            parameter_bindings: Box::new([]),
        };
        let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 64 };
        let kir = build_kir_correlation_index(body, 4, &mut budget).unwrap();
        let sites = index_semantic_access_sites(&correspondence, &kir, &mut budget).unwrap();

        assert_eq!(kir.memory_consumers.len(), 2);
        assert!(kir.unmodeled_memory_effects.is_empty());
        for ordinal in 0..2 {
            assert_eq!(
                sites.get(&(FunctionOperationLocation::new(BlockId(0), 0), ordinal,)),
                Some(&SemanticAccessSiteV1 {
                    block: 0,
                    statement: Some(0),
                    ordinal,
                })
            );
        }
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_source_effect_reordering() {
        let fixture = unsupported_index_correlation_fixture();
        let mut module = fixture.module.clone();
        module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .push(Operation::effect_free(
                ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ));
        verify_module(&module).expect("two-effect fixture remains valid");
        let mut correspondence = fixture.correspondence.clone();
        correspondence.statement_operation_spans[0].operation_count = 6;
        correspondence.terminator_operation_spans[0].first_operation_ordinal = 6;
        let lowering =
            ranked_correlation_input_for_accesses(&[AccessKindAttr::Read, AccessKindAttr::Read], 1);
        let sources = [
            ProductionRankedAccessSourceV1::new(0, Some(0), 0, 0, 4),
            ProductionRankedAccessSourceV1::new(0, Some(0), 1, 0, 3),
        ];
        assert!(matches!(
            validate_mir_pliron_translation_v1(&module, &correspondence, &lowering, &sources, 16,),
            Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                first_semantic_block: 0,
                second_semantic_block: 0,
                ..
            })
        ));
    }

    fn ranked_correlation_input_with_barrier() -> ProductionRankedKernelLoweringInputV1 {
        let view = ProductionRankedValueIdV1::new(0);
        let index = ProductionRankedValueIdV1::new(1);
        let kernel = ProductionRankedKernelV1::new(
            "unsupported_index_correlation",
            0,
            vec![ProductionRankedBlockV1::new(
                vec![
                    ProductionRankedOperationV1::ExecutionLayout {
                        grid_identity: 1,
                        global_extents: [64, 1, 1],
                        workgroup_extents: [64, 1, 1],
                        subgroup_size: 64,
                        full_physical_workgroups: true,
                    },
                    ProductionRankedOperationV1::ViewInSpace {
                        result: view,
                        element_width: 32,
                        writable: false,
                        shape: vec![64],
                        dynamic_extents: vec![],
                        memory_space: dialect_kernel::MemorySpaceAttr::Global,
                        allocation_origin: 1,
                        noalias_class: 1,
                    },
                    ProductionRankedOperationV1::InvocationIndex {
                        result: index,
                        dimension: 0,
                        launch_extent: 64,
                    },
                    ProductionRankedOperationV1::Access {
                        kind: AccessKindAttr::Read,
                        view: ProductionRankedValueV1::Local(view),
                        indices: vec![ProductionRankedValueV1::Local(index)],
                    },
                    ProductionRankedOperationV1::Barrier {
                        execution_scope: dialect_gpu::HierarchyAttr::Workgroup,
                        memory_scope: dialect_gpu::MemoryScopeAttr::Workgroup,
                        address_space: dialect_gpu::AddressSpaceAttr::Workgroup,
                        order: dialect_gpu::MemoryOrderAttr::AcquireRelease,
                    },
                ],
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .expect("barrier translation fixture must be structurally valid");
        compile_ranked_kernel_for_lowering_v1(
            ProductionConstructionV1::ranked_kernel("barrier_translation_module", kernel).unwrap(),
            ProductionSessionLimitsV1::default(),
        )
        .expect("barrier translation fixture must pass mandatory checks")
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_synchronization_substitution() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input_with_barrier();
        assert!(matches!(
            validate_translation_fixture(&fixture, &lowering, &[fixture.source], 16),
            Err(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch)
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_divergent_tensor_contracts() {
        let contract = TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64();
        let divergent = ProductionRankedKernelV1::new(
            "tensor_translation",
            0,
            vec![ProductionRankedBlockV1::new(
                vec![ProductionRankedOperationV1::TensorLayout {
                    contract,
                    convergence: dialect_kernel::TensorConvergenceAttr::Divergent,
                    active_lanes: 64,
                    binding: None,
                }],
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .expect("divergent tensor recipe remains structurally representable");
        assert!(matches!(
            ranked_tensor_contracts_v1(&divergent),
            Err(ProductionMirPlironTranslationErrorV1::TensorContractMismatch)
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_enforces_exact_resource_boundary() {
        let fixture = unsupported_index_correlation_fixture();
        let lowering = ranked_correlation_input(AccessKindAttr::Read, 1);
        assert!(validate_translation_fixture(&fixture, &lowering, &[fixture.source], 5,).is_ok());
        assert!(matches!(
            validate_translation_fixture(&fixture, &lowering, &[fixture.source], 4),
            Err(ProductionMirPlironTranslationErrorV1::ResourceLimit)
        ));
    }

    struct ValueTranslationFixtureV1 {
        module: Module,
        correspondence: SemanticKirCorrespondenceV1,
        lowering: ProductionRankedKernelLoweringInputV1,
        sources: [ProductionRankedAccessSourceV1; 2],
    }

    fn value_translation_fixture(
        operation: ProductionSemanticBinaryOpV2,
        constant_bits: u64,
    ) -> ValueTranslationFixtureV1 {
        let scalar = ProductionSemanticScalarTypeV2::Float { bits: 32 };
        let slice = Type::slice(
            Type::Scalar(ScalarType::F32),
            AddressSpace::Global,
            AccessMode::ReadWrite,
        );
        let pointer = Type::pointer(
            Type::Scalar(ScalarType::F32),
            AddressSpace::Global,
            AccessMode::ReadWrite,
        );
        let mut block = BasicBlock::new(BlockId(0));
        block.operations = vec![
            Operation::effect_free(
                ValueDef::new(ValueId(1), Type::INDEX),
                OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(2), Type::INDEX),
                OperationKind::Binary {
                    op: BinaryOp::Multiply,
                    lhs: ValueId(1),
                    rhs: ValueId(1),
                },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(3), pointer.clone()),
                OperationKind::SliceData { slice: ValueId(0) },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(4), pointer),
                OperationKind::GetElementPointer {
                    base: ValueId(3),
                    offset: ValueId(2),
                },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(5), Type::Scalar(ScalarType::F32)),
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
                OperationKind::Constant(Constant::F32Bits(0x3f80_0000)),
            ),
            Operation::effect_free(
                ValueDef::new(ValueId(7), Type::Scalar(ScalarType::F32)),
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: ValueId(5),
                    rhs: ValueId(6),
                },
            ),
            Operation::new(
                vec![],
                OperationKind::Store {
                    pointer: ValueId(4),
                    value: ValueId(7),
                    access: MemoryAccess::new(AddressSpace::Global, 4),
                },
            ),
        ];
        block.terminator = Some(Terminator::Return { values: vec![] });
        let mut module = Module::new("value-translation");
        module.functions.push(Function::kernel_entry(
            "value_translation",
            Signature::new(vec![slice], vec![]),
            vec![ValueId(0)],
            vec![block],
        ));
        module.kernels.push(Kernel::new(
            "value-translation",
            "value_translation",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(64),
            },
        ));
        verify_module(&module).expect("value translation Kernel IR must verify");
        let correspondence = SemanticKirCorrespondenceV1 {
            semantic_sha256: [8; 32],
            function_count: 1,
            blocks: vec![SemanticKirBlockCorrespondenceV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                source_statement_count: 1,
            }]
            .into_boxed_slice(),
            statement_operation_spans: vec![SemanticKirStatementOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                statement_ordinal: 0,
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 0,
                operation_count: 8,
            }]
            .into_boxed_slice(),
            terminator_operation_spans: vec![SemanticKirTerminatorOperationSpanV1 {
                semantic_function: SemanticFunctionIdV1::from_index(0),
                semantic_block: SemanticBlockIdV1::from_index(0),
                kernel_ir_block: BlockId(0),
                first_operation_ordinal: 8,
                operation_count: 0,
            }]
            .into_boxed_slice(),
            synthetic_operation_spans: Box::new([]),
            parameter_bindings: Box::new([]),
        };

        let view = ProductionRankedValueIdV1::new(0);
        let index = ProductionRankedValueIdV1::new(1);
        let expression_id = ProductionRankedValueIdV1::new(2);
        let load = fe2o3_pliron::ProductionSemanticLoadV2 {
            block: 0,
            operation: 3,
            scalar,
            allocation_origin: 1,
            view: ProductionRankedValueV1::Local(view),
            indices: vec![ProductionRankedValueV1::Local(index)].into_boxed_slice(),
        };
        let expression = ProductionSemanticExpressionV2::Binary {
            operation,
            scalar,
            overflow: ProductionOverflowContractV2::Wrapping,
            lhs: Box::new(ProductionSemanticExpressionV2::Load(load)),
            rhs: Box::new(ProductionSemanticExpressionV2::Constant {
                scalar,
                bits: constant_bits,
            }),
        };
        let numerical_contract = ProductionNumericalContractV2::exact_for_expression(&expression);
        let kernel = ProductionRankedKernelV1::new(
            "value_translation",
            0,
            vec![ProductionRankedBlockV1::new(
                vec![
                    ProductionRankedOperationV1::ExecutionLayout {
                        grid_identity: 1,
                        global_extents: [64, 1, 1],
                        workgroup_extents: [64, 1, 1],
                        subgroup_size: 64,
                        full_physical_workgroups: true,
                    },
                    ProductionRankedOperationV1::ViewInSpace {
                        result: view,
                        element_width: 32,
                        writable: true,
                        shape: vec![64],
                        dynamic_extents: vec![],
                        memory_space: dialect_kernel::MemorySpaceAttr::Global,
                        allocation_origin: 1,
                        noalias_class: 1,
                    },
                    ProductionRankedOperationV1::InvocationIndex {
                        result: index,
                        dimension: 0,
                        launch_extent: 64,
                    },
                    ProductionRankedOperationV1::Access {
                        kind: AccessKindAttr::Read,
                        view: ProductionRankedValueV1::Local(view),
                        indices: vec![ProductionRankedValueV1::Local(index)],
                    },
                    ProductionRankedOperationV1::SemanticExpression {
                        result: expression_id,
                        expression,
                        numerical_contract,
                    },
                    ProductionRankedOperationV1::ValueAccess {
                        kind: AccessKindAttr::Write,
                        view: ProductionRankedValueV1::Local(view),
                        indices: vec![ProductionRankedValueV1::Local(index)],
                        value: ProductionRankedValueV1::Local(expression_id),
                    },
                ],
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .expect("value translation ranked kernel must be valid");
        let lowering = compile_ranked_kernel_for_lowering_v1(
            ProductionConstructionV1::ranked_kernel("value_translation_module", kernel).unwrap(),
            ProductionSessionLimitsV1::default(),
        )
        .expect("value translation ranked kernel must pass mandatory checks");
        ValueTranslationFixtureV1 {
            module,
            correspondence,
            lowering,
            sources: [
                ProductionRankedAccessSourceV1::new(0, Some(0), 0, 0, 3),
                ProductionRankedAccessSourceV1::new(0, Some(0), 1, 0, 5),
            ],
        }
    }

    #[test]
    fn mir_pliron_translation_validation_reconstructs_exact_write_expression() {
        let fixture = value_translation_fixture(
            ProductionSemanticBinaryOpV2::Add,
            u64::from(0x3f80_0000_u32),
        );
        let report = validate_mir_pliron_translation_v1(
            &fixture.module,
            &fixture.correspondence,
            &fixture.lowering,
            &fixture.sources,
            32,
        )
        .expect("the independently reconstructed write expression must agree");
        assert_eq!(report.memory_effects(), 2);
        assert_eq!(report.value_expressions(), 1);
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_operator_and_constant_drift() {
        for fixture in [
            value_translation_fixture(
                ProductionSemanticBinaryOpV2::Subtract,
                u64::from(0x3f80_0000_u32),
            ),
            value_translation_fixture(
                ProductionSemanticBinaryOpV2::Add,
                u64::from(0x4000_0000_u32),
            ),
        ] {
            assert!(matches!(
                validate_mir_pliron_translation_v1(
                    &fixture.module,
                    &fixture.correspondence,
                    &fixture.lowering,
                    &fixture.sources,
                    32,
                ),
                Err(
                    ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch {
                        location: FunctionOperationLocation {
                            block: BlockId(0),
                            operation_index: 7,
                        },
                    }
                )
            ));
        }
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_executable_value_drift() {
        let mut fixture = value_translation_fixture(
            ProductionSemanticBinaryOpV2::Add,
            u64::from(0x3f80_0000_u32),
        );
        let OperationKind::Binary { op, .. } =
            &mut fixture.module.functions[0].body.as_mut().unwrap().blocks[0].operations[6].kind
        else {
            unreachable!();
        };
        *op = BinaryOp::Multiply;
        verify_module(&fixture.module).expect("mutated arithmetic remains valid Kernel IR");
        assert!(matches!(
            validate_mir_pliron_translation_v1(
                &fixture.module,
                &fixture.correspondence,
                &fixture.lowering,
                &fixture.sources,
                32,
            ),
            Err(ProductionMirPlironTranslationErrorV1::ValueExpressionMismatch { .. })
        ));
    }

    #[test]
    fn mir_pliron_translation_validation_rejects_effect_control_flow_reversal() {
        let mut fixture = unsupported_index_correlation_fixture();
        fixture.module.functions[0].body.as_mut().unwrap().blocks[0].terminator =
            Some(Terminator::Branch {
                target: BlockId(1),
                arguments: vec![],
            });
        let mut second = BasicBlock::new(BlockId(1));
        second.operations.push(Operation::effect_free(
            ValueDef::new(ValueId(6), Type::Scalar(ScalarType::F32)),
            OperationKind::Load {
                pointer: ValueId(4),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ));
        second.terminator = Some(Terminator::Return { values: vec![] });
        fixture.module.functions[0]
            .body
            .as_mut()
            .unwrap()
            .blocks
            .push(second);
        verify_module(&fixture.module).expect("two-block Kernel IR must remain valid");

        let mut blocks = fixture.correspondence.blocks.to_vec();
        blocks.push(SemanticKirBlockCorrespondenceV1 {
            semantic_function: SemanticFunctionIdV1::from_index(0),
            semantic_block: SemanticBlockIdV1::from_index(1),
            kernel_ir_block: BlockId(1),
            source_statement_count: 1,
        });
        fixture.correspondence.blocks = blocks.into_boxed_slice();
        let mut statements = fixture.correspondence.statement_operation_spans.to_vec();
        statements.push(SemanticKirStatementOperationSpanV1 {
            semantic_function: SemanticFunctionIdV1::from_index(0),
            semantic_block: SemanticBlockIdV1::from_index(1),
            statement_ordinal: 0,
            kernel_ir_block: BlockId(1),
            first_operation_ordinal: 0,
            operation_count: 1,
        });
        fixture.correspondence.statement_operation_spans = statements.into_boxed_slice();
        let mut terminators = fixture.correspondence.terminator_operation_spans.to_vec();
        terminators.push(SemanticKirTerminatorOperationSpanV1 {
            semantic_function: SemanticFunctionIdV1::from_index(0),
            semantic_block: SemanticBlockIdV1::from_index(1),
            kernel_ir_block: BlockId(1),
            first_operation_ordinal: 1,
            operation_count: 0,
        });
        fixture.correspondence.terminator_operation_spans = terminators.into_boxed_slice();

        let view = ProductionRankedValueIdV1::new(0);
        let index = ProductionRankedValueIdV1::new(1);
        let access = || ProductionRankedOperationV1::Access {
            kind: AccessKindAttr::Read,
            view: ProductionRankedValueV1::Local(view),
            indices: vec![ProductionRankedValueV1::Local(index)],
        };
        let kernel = ProductionRankedKernelV1::new(
            "control_flow_reversal",
            0,
            vec![
                ProductionRankedBlockV1::new(
                    vec![
                        ProductionRankedOperationV1::ExecutionLayout {
                            grid_identity: 1,
                            global_extents: [64, 1, 1],
                            workgroup_extents: [64, 1, 1],
                            subgroup_size: 64,
                            full_physical_workgroups: true,
                        },
                        ProductionRankedOperationV1::ViewInSpace {
                            result: view,
                            element_width: 32,
                            writable: false,
                            shape: vec![64],
                            dynamic_extents: vec![],
                            memory_space: dialect_kernel::MemorySpaceAttr::Global,
                            allocation_origin: 1,
                            noalias_class: 1,
                        },
                        ProductionRankedOperationV1::InvocationIndex {
                            result: index,
                            dimension: 0,
                            launch_extent: 64,
                        },
                        access(),
                    ],
                    ProductionRankedTerminatorV1::Branch { target: 1 },
                ),
                ProductionRankedBlockV1::new(vec![access()], ProductionRankedTerminatorV1::Return),
            ],
        )
        .expect("reversed effect CFG remains structurally valid ranked IR");
        let lowering = compile_ranked_kernel_for_lowering_v1(
            ProductionConstructionV1::ranked_kernel("control_flow_reversal_module", kernel)
                .unwrap(),
            ProductionSessionLimitsV1::default(),
        )
        .expect("reversed effect CFG passes local safety checks before translation validation");
        let sources = [
            ProductionRankedAccessSourceV1::new(0, Some(0), 0, 1, 0),
            ProductionRankedAccessSourceV1::new(1, Some(0), 0, 0, 3),
        ];
        assert!(matches!(
            validate_mir_pliron_translation_v1(
                &fixture.module,
                &fixture.correspondence,
                &lowering,
                &sources,
                32,
            ),
            Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                first_semantic_block: 0,
                second_semantic_block: 1,
                ..
            }) | Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                first_semantic_block: 1,
                second_semantic_block: 0,
                ..
            })
        ));
    }

    #[test]
    fn rust_usize_and_kernel_index_are_edge_transport_equivalent() {
        let index = Type::INDEX;
        let u64_type = Type::Scalar(ScalarType::U64);
        assert!(index_and_u64_are_transport_equivalent(&index, &u64_type));
        assert!(index_and_u64_are_transport_equivalent(&u64_type, &index));
        assert!(!index_and_u64_are_transport_equivalent(
            &index,
            &Type::Scalar(ScalarType::U32),
        ));
        assert!(!index_and_u64_are_transport_equivalent(
            &Type::Scalar(ScalarType::U64),
            &Type::Scalar(ScalarType::I64),
        ));
    }
}
