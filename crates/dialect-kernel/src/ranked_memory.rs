use std::{error::Error, fmt};

use pliron::{
    builtin::{
        ATTR_KEY_DEBUG_INFO,
        op_interfaces::{IsTerminatorInterface, NRegionsInterface, NResultsInterface},
        ops::FuncOp,
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{pliron_attr, pliron_op, pliron_type},
    op::Op,
    operation::Operation,
    result::Result as PlironResult,
    r#type::{Type, Typed, TypedHandle},
    value::Value,
    verify_err, verify_err_noloc,
};

/// Maximum rank admitted by target-neutral ranked memory operations.
pub const MAX_RANKED_MEMORY_RANK: usize = 8;

/// Maximum explicit dependency count admitted by one deterministic summary.
pub const MAX_DETERMINISTIC_JOIN_INPUTS_V1: usize = 64;

/// A zero extent in [`RankedViewType`] denotes a runtime dimension.
pub const DYNAMIC_EXTENT: u64 = 0;

/// Target-neutral scalar element widths supported by the first ranked-memory schema.
pub const SUPPORTED_ELEMENT_WIDTHS: [u32; 5] = [8, 16, 32, 64, 128];

/// Construction or local verification failure for ranked-memory IR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RankedMemoryError {
    RankOutOfBounds(usize),
    UnsupportedElementWidth(u32),
    DynamicExtentCountMismatch { expected: usize, actual: usize },
    OperandCountMismatch { expected: usize, actual: usize },
    ForeignViewType,
    ForeignIndexType { operand: usize },
    DimensionOutOfBounds { dimension: u32, rank: usize },
    WriteThroughReadOnlyView,
    MissingAtomicContract,
    UnexpectedAtomicContract,
    MalformedPayload(&'static str),
}

impl fmt::Display for RankedMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankOutOfBounds(rank) => write!(
                formatter,
                "ranked memory rank {rank} is outside 1..={MAX_RANKED_MEMORY_RANK}"
            ),
            Self::UnsupportedElementWidth(width) => write!(
                formatter,
                "ranked memory element width {width} is not one of {SUPPORTED_ELEMENT_WIDTHS:?}"
            ),
            Self::DynamicExtentCountMismatch { expected, actual } => write!(
                formatter,
                "ranked view requires {expected} dynamic extents but has {actual}"
            ),
            Self::OperandCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "operation requires {expected} operands but has {actual}"
                )
            }
            Self::ForeignViewType => formatter.write_str("operand 0 is not a kernel ranked view"),
            Self::ForeignIndexType { operand } => {
                write!(formatter, "operand {operand} is not a kernel index")
            }
            Self::DimensionOutOfBounds { dimension, rank } => write!(
                formatter,
                "dimension {dimension} is outside ranked view rank {rank}"
            ),
            Self::WriteThroughReadOnlyView => {
                formatter.write_str("write access uses a read-only ranked view")
            }
            Self::MissingAtomicContract => {
                formatter.write_str("atomic access requires explicit ordering and scope")
            }
            Self::UnexpectedAtomicContract => {
                formatter.write_str("non-atomic access cannot carry an atomic contract")
            }
            Self::MalformedPayload(message) => formatter.write_str(message),
        }
    }
}

impl Error for RankedMemoryError {}

fn check_shape(shape: &[u64]) -> Result<(), RankedMemoryError> {
    if !(1..=MAX_RANKED_MEMORY_RANK).contains(&shape.len()) {
        return Err(RankedMemoryError::RankOutOfBounds(shape.len()));
    }
    Ok(())
}

fn check_element_width(element_width: u32) -> Result<(), RankedMemoryError> {
    if SUPPORTED_ELEMENT_WIDTHS.contains(&element_width) {
        Ok(())
    } else {
        Err(RankedMemoryError::UnsupportedElementWidth(element_width))
    }
}

/// A target-neutral ranked view. Zero extents are dynamic; nonzero extents are static.
#[pliron_type(
    name = "kernel.ranked_view",
    format = "`<` $element_width `,` $writable `,` `[` vec($shape, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RankedViewType {
    element_width: u32,
    writable: bool,
    shape: Vec<u64>,
}

impl RankedViewType {
    pub fn new(
        context: &Context,
        element_width: u32,
        writable: bool,
        shape: Vec<u64>,
    ) -> Result<TypedHandle<Self>, RankedMemoryError> {
        check_element_width(element_width)?;
        check_shape(&shape)?;
        Ok(Self::instantiate(
            Self {
                element_width,
                writable,
                shape,
            },
            context,
        ))
    }

    pub const fn element_width(&self) -> u32 {
        self.element_width
    }

    pub const fn writable(&self) -> bool {
        self.writable
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dynamic_extent_count(&self) -> usize {
        self.shape
            .iter()
            .filter(|extent| **extent == DYNAMIC_EXTENT)
            .count()
    }
}

impl Verify for RankedViewType {
    fn verify(&self, _context: &Context) -> PlironResult<()> {
        if let Err(error) =
            check_element_width(self.element_width).and_then(|()| check_shape(&self.shape))
        {
            return verify_err_noloc!(error);
        }
        Ok(())
    }
}

/// Unsigned target-neutral index type. Non-negativity is intrinsic to the type.
#[pliron_type(name = "kernel.index", format, generate_get = true, verifier = "succ")]
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct IndexType;

/// Opaque structural obligation carrier for one checked-access condition.
///
/// The type is intentionally crate-private. Its presence proves no source
/// correspondence or compiler authority by itself. Raw IR can spell the type;
/// production analysis must separately bind it to an owner-held semantic-MIR
/// success path.
#[pliron_type(
    name = "kernel.checked_access_capability",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct CheckedAccessCapabilityType;

/// Constant index payload.
#[pliron_attr(name = "kernel.index_value", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IndexValueAttr(pub u64);

impl IndexValueAttr {
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Ranked-view dimension selector.
#[pliron_attr(name = "kernel.dimension", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DimensionAttr(pub u32);

impl DimensionAttr {
    pub const fn dimension(self) -> u32 {
        self.0
    }
}

/// Whether an indexed access reads or writes the view.
#[pliron_attr(name = "kernel.access_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessKindAttr {
    Read,
    Write,
    AtomicRead,
    AtomicWrite,
    AtomicReadModifyWrite,
}

/// Coverage expected from the writes associated with an ownership contract.
///
/// The attribute describes the obligation only. Whole-function analysis must
/// derive the owners from actual `kernel.access` operations.
#[pliron_attr(name = "kernel.ownership_coverage", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OwnershipCoverageAttr {
    /// Every logical element of the ranked view has exactly one invocation
    /// owner after guarded control flow is applied.
    ExactView,
    /// Every coordinate in the compiler-derived set of actual writes has one
    /// owner. This proves disjoint effect ownership without claiming that the
    /// writes cover storage outside the kernel's effect domain.
    ExactEffectDomain,
    /// Every logical element of the ranked view has exactly one observable
    /// write event, that event is final, and every observable global write in
    /// the function is covered by an ownership contract.
    TotalView,
    /// Every physical invocation contributes exactly once through an atomic
    /// write to the ranked view. This proves contribution coverage only; it
    /// does not prove an operator identity, associativity, or the final value.
    CollectiveContributions,
}

/// Shape requirement for hierarchy-level ownership summaries.
#[pliron_attr(name = "kernel.ownership_partition", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OwnershipPartitionAttr {
    /// Subgroup and workgroup regions may be arbitrary exact sets.
    ExactSets,
    /// Every nonempty subgroup and workgroup region must be a dense rectangle.
    DenseRectangles,
}

/// Ordering requested by one target-neutral atomic access.
#[pliron_attr(name = "kernel.atomic_ordering", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AtomicOrderingAttr {
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

/// Visibility scope requested by one target-neutral atomic access.
#[pliron_attr(name = "kernel.atomic_scope", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AtomicScopeAttr {
    SingleThread,
    Workgroup,
    /// All workgroups executing on one GPU agent. This covers distinct
    /// workgroups within the same retained grid identity.
    Agent,
    /// Device-wide scope retained by source languages that distinguish it
    /// from the HSA agent scope. It is at least as wide as `Agent`.
    Device,
    System,
}

impl AtomicScopeAttr {
    pub const fn rank(self) -> u8 {
        match self {
            Self::SingleThread => 0,
            Self::Workgroup => 1,
            Self::Agent => 2,
            Self::Device => 3,
            Self::System => 4,
        }
    }
}

impl AccessKindAttr {
    pub const fn is_atomic(self) -> bool {
        matches!(
            self,
            Self::AtomicRead | Self::AtomicWrite | Self::AtomicReadModifyWrite
        )
    }

    pub const fn reads_memory(self) -> bool {
        matches!(
            self,
            Self::Read | Self::AtomicRead | Self::AtomicReadModifyWrite
        )
    }

    pub const fn writes_memory(self) -> bool {
        matches!(
            self,
            Self::Write | Self::AtomicWrite | Self::AtomicReadModifyWrite
        )
    }
}

/// Storage domain used by the target-neutral concurrent-effect analyses.
#[pliron_attr(name = "kernel.memory_space", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemorySpaceAttr {
    Private,
    Workgroup,
    Global,
}

/// Stable compiler-issued identity for the source allocation behind a view.
/// Zero is the fail-closed unknown origin used by legacy or unauthenticated IR.
#[pliron_attr(name = "kernel.allocation_origin", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AllocationOriginAttr(pub u64);

impl AllocationOriginAttr {
    pub const fn identity(self) -> u64 {
        self.0
    }
}

/// Compiler-issued no-alias partition for ranked allocations.
///
/// Zero may alias every partition. Distinct nonzero classes are proven
/// disjoint; views in the same class remain conservatively may-alias.
#[pliron_attr(name = "kernel.noalias_class", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NoAliasClassAttr(pub u64);

impl NoAliasClassAttr {
    pub const fn identity(self) -> u64 {
        self.0
    }
}

/// One logical launch dimension selected by [`InvocationIndexOp`].
#[pliron_attr(name = "kernel.invocation_dimension", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InvocationDimensionAttr(pub u32);

impl InvocationDimensionAttr {
    pub const fn dimension(self) -> u32 {
        self.0
    }
}

/// Static launch extent. Zero denotes a runtime extent that analyses must
/// prove from another retained fact or reject as unresolved.
#[pliron_attr(name = "kernel.launch_extent", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LaunchExtentAttr(pub u64);

impl LaunchExtentAttr {
    pub const fn extent(self) -> u64 {
        self.0
    }
}

/// Number of leading analysis-split operands that completely describe its
/// controlling predicate dependencies. Remaining operands are successor data.
#[pliron_attr(
    name = "kernel.analysis_split_control_count",
    format = "$0",
    verifier = "succ"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AnalysisSplitControlCountAttr(pub u32);

impl AnalysisSplitControlCountAttr {
    pub const fn count(self) -> u32 {
        self.0
    }
}

/// Closed arithmetic supported by sparse index analysis.
#[pliron_attr(name = "kernel.index_binary_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndexBinaryKindAttr {
    Add,
    Multiply,
    Remainder,
    Divide,
}

/// Materializes a ranked view. Its operands are the runtime extents in shape order.
#[pliron_op(
    name = "kernel.ranked_view",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (
        kernel_memory_space: MemorySpaceAttr,
        kernel_allocation_origin: AllocationOriginAttr,
        kernel_noalias_class: NoAliasClassAttr
    )
)]
pub struct RankedViewOp;

impl RankedViewOp {
    pub fn new(
        context: &mut Context,
        view_type: TypedHandle<RankedViewType>,
        dynamic_extents: Vec<Value>,
    ) -> Result<Self, RankedMemoryError> {
        Self::new_in_space(context, view_type, dynamic_extents, MemorySpaceAttr::Global)
    }

    pub fn new_in_space(
        context: &mut Context,
        view_type: TypedHandle<RankedViewType>,
        dynamic_extents: Vec<Value>,
        memory_space: MemorySpaceAttr,
    ) -> Result<Self, RankedMemoryError> {
        Self::new_in_space_with_allocation_contract(
            context,
            view_type,
            dynamic_extents,
            memory_space,
            0,
            0,
        )
    }

    pub fn new_in_space_with_allocation_contract(
        context: &mut Context,
        view_type: TypedHandle<RankedViewType>,
        dynamic_extents: Vec<Value>,
        memory_space: MemorySpaceAttr,
        allocation_origin: u64,
        noalias_class: u64,
    ) -> Result<Self, RankedMemoryError> {
        let expected = view_type.deref(context).dynamic_extent_count();
        if dynamic_extents.len() != expected {
            return Err(RankedMemoryError::DynamicExtentCountMismatch {
                expected,
                actual: dynamic_extents.len(),
            });
        }
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![view_type.into()],
            dynamic_extents,
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_memory_space(context, memory_space);
        op.set_attr_kernel_allocation_origin(context, AllocationOriginAttr(allocation_origin));
        op.set_attr_kernel_noalias_class(context, NoAliasClassAttr(noalias_class));
        Ok(op)
    }

    pub fn view_type(&self, context: &Context) -> Option<TypedHandle<RankedViewType>> {
        TypedHandle::from_handle(self.get_operation().deref(context).get_type(0), context).ok()
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }

    pub fn memory_space(&self, context: &Context) -> Option<MemorySpaceAttr> {
        self.get_attr_kernel_memory_space(context)
            .map(|space| *space)
    }

    pub fn allocation_origin(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_allocation_origin(context)
            .map(|origin| origin.identity())
            .or(Some(0))
    }

    pub fn noalias_class(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_noalias_class(context)
            .map(|class| class.identity())
            .or(Some(0))
    }

    /// Returns the runtime extent bound to one dynamic shape dimension.
    /// Static dimensions and out-of-rank selectors return `None`.
    pub fn dynamic_extent(&self, context: &Context, dimension: usize) -> Option<Value> {
        let view_type = self.view_type(context)?;
        let view_type = view_type.deref(context);
        if view_type.shape().get(dimension).copied()? != DYNAMIC_EXTENT {
            return None;
        }
        let operand = view_type.shape()[..dimension]
            .iter()
            .filter(|extent| **extent == DYNAMIC_EXTENT)
            .count();
        Some(self.get_operation().deref(context).get_operand(operand))
    }
}

impl Verify for RankedViewOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let Some(view_type) = self.view_type(context) else {
            return verify_err!(self.loc(context), RankedMemoryError::ForeignViewType);
        };
        let expected = view_type.deref(context).dynamic_extent_count();
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let has_origin = self.get_attr_kernel_allocation_origin(context).is_some();
        let has_class = self.get_attr_kernel_noalias_class(context).is_some();
        if operation.get_num_operands() != expected
            || !matches!(payload_attribute_count(&operation), 1 | 3)
            || has_origin != has_class
            || self.memory_space(context).is_none()
            || self.allocation_origin(context).is_none()
            || self.noalias_class(context).is_none()
            || self
                .noalias_class(context)
                .is_some_and(|class| class != 0 && self.allocation_origin(context) == Some(0))
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::DynamicExtentCountMismatch {
                    expected,
                    actual: operation.get_num_operands(),
                }
            );
        }
        for operand in 0..expected {
            require_index_operand(self, context, operand)?;
        }
        Ok(())
    }
}

/// Produces an unsigned constant index.
#[pliron_op(
    name = "kernel.index_constant",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (kernel_index_value: IndexValueAttr)
)]
pub struct IndexConstantOp;

impl IndexConstantOp {
    pub fn new(context: &mut Context, value: u64) -> Self {
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_index_value(context, IndexValueAttr(value));
        op
    }

    pub fn value(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_index_value(context)
            .map(|value| value.value())
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }
}

impl Verify for IndexConstantOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        if self.get_operation().deref(context).get_num_operands() != 0
            || self.value(context).is_none()
            || payload_attribute_count(&self.get_operation().deref(context)) != 1
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.index_constant has malformed payload")
            );
        }
        Ok(())
    }
}

/// Produces an index whose source provenance is unavailable to analysis.
#[pliron_op(
    name = "kernel.index_unknown",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>]
)]
pub struct IndexUnknownOp;

impl IndexUnknownOp {
    pub fn new(context: &mut Context) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![],
            vec![],
            0,
        ))
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }
}

impl Verify for IndexUnknownOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        if self.get_operation().deref(context).get_num_operands() != 0
            || payload_attribute_count(&self.get_operation().deref(context)) != 0
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.index_unknown has malformed payload")
            );
        }
        Ok(())
    }
}

/// Produces the logical invocation coordinate in one launch dimension.
#[pliron_op(
    name = "kernel.invocation_index",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (
        kernel_invocation_dimension: InvocationDimensionAttr,
        kernel_launch_extent: LaunchExtentAttr
    )
)]
pub struct InvocationIndexOp;

impl InvocationIndexOp {
    pub fn new(context: &mut Context, dimension: u32, launch_extent: u64) -> Self {
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_invocation_dimension(context, InvocationDimensionAttr(dimension));
        op.set_attr_kernel_launch_extent(context, LaunchExtentAttr(launch_extent));
        op
    }

    pub fn dimension(&self, context: &Context) -> Option<u32> {
        self.get_attr_kernel_invocation_dimension(context)
            .map(|dimension| dimension.dimension())
    }

    pub fn launch_extent(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_launch_extent(context)
            .map(|extent| extent.extent())
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }
}

impl Verify for InvocationIndexOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let raw = self.get_operation();
        let raw = raw.deref(context);
        if raw.get_num_operands() != 0
            || payload_attribute_count(&raw) != 2
            || self
                .dimension(context)
                .is_none_or(|dimension| dimension as usize >= MAX_RANKED_MEMORY_RANK)
            || self.launch_extent(context).is_none()
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.invocation_index has malformed dimension, extent, or result"
                )
            );
        }
        Ok(())
    }
}

/// Explicit unsigned-width conversion for one ranked index value.
///
/// The result is the low `bit_width` bits of `source`, interpreted as an
/// unsigned index. Its inclusive upper bound is therefore derived from the
/// operation semantics rather than from an assumption about `source`.
#[pliron_op(
    name = "kernel.index_unsigned_cast",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (kernel_unsigned_bit_width: IndexValueAttr)
)]
pub struct IndexUnsignedCastOp;

impl IndexUnsignedCastOp {
    pub fn new(context: &mut Context, value: Value, bit_width: u64) -> Self {
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![value],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_unsigned_bit_width(context, IndexValueAttr(bit_width));
        op
    }

    pub fn source(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }

    pub fn bit_width(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_unsigned_bit_width(context)
            .map(|width| width.value())
    }

    pub fn inclusive_upper_bound(&self, context: &Context) -> Option<u64> {
        match self.bit_width(context)? {
            8 => Some(u8::MAX.into()),
            16 => Some(u16::MAX.into()),
            32 => Some(u32::MAX.into()),
            64 => Some(u64::MAX),
            _ => None,
        }
    }

    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

impl Verify for IndexUnsignedCastOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        if operation.get_num_operands() != 1
            || payload_attribute_count(&operation) != 1
            || self.inclusive_upper_bound(context).is_none()
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.index_unsigned_cast requires one index source, one index result, and an unsigned width in {8, 16, 32, 64}",
                )
            );
        }
        require_index_operand(self, context, 0)?;
        if !is_index_type(self.result(context), context) {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.index_unsigned_cast result must be kernel.index",
                )
            );
        }
        Ok(())
    }
}

/// Target-neutral unsigned index arithmetic retained for sparse analysis.
///
/// Analyses may only use order-preserving affine reasoning when they prove the
/// operation cannot overflow over the retained domain. Unproved overflow is
/// not a source of control-uniformity or memory-safety authority.
#[pliron_op(
    name = "kernel.index_binary",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (kernel_index_binary_kind: IndexBinaryKindAttr)
)]
pub struct IndexBinaryOp;

impl IndexBinaryOp {
    pub fn new(context: &mut Context, kind: IndexBinaryKindAttr, lhs: Value, rhs: Value) -> Self {
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_index_binary_kind(context, kind);
        op
    }

    pub fn kind(&self, context: &Context) -> Option<IndexBinaryKindAttr> {
        self.get_attr_kernel_index_binary_kind(context)
            .map(|kind| *kind)
    }

    pub fn lhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn rhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(1)
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }
}

impl Verify for IndexBinaryOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let raw = self.get_operation();
        let raw = raw.deref(context);
        if raw.get_num_operands() != 2
            || payload_attribute_count(&raw) != 1
            || self.kind(context).is_none()
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.index_binary has malformed payload")
            );
        }
        require_index_operand(self, context, 0)?;
        require_index_operand(self, context, 1)
    }
}

/// Abstract result of a total deterministic computation over explicit inputs.
///
/// This operation records dependency only. It carries no source identity,
/// uniformity declaration, compiler refinement, artifact, or launch authority.
#[pliron_op(
    name = "kernel.deterministic_join",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>]
)]
pub struct DeterministicJoinOp;

impl DeterministicJoinOp {
    pub fn new(context: &mut Context, dependencies: Vec<Value>) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            dependencies,
            vec![],
            0,
        ))
    }

    pub fn dependencies(&self, context: &Context) -> Vec<Value> {
        self.get_operation().deref(context).operands().collect()
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }

    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

impl Verify for DeterministicJoinOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let count = operation.get_num_operands();
        if !(1..=MAX_DETERMINISTIC_JOIN_INPUTS_V1).contains(&count)
            || payload_attribute_count(&operation) != 0
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.deterministic_join has malformed or unbounded dependencies",
                )
            );
        }
        for operand in 0..count {
            require_index_operand(self, context, operand)?;
        }
        Ok(())
    }
}

/// Checked row-major index mapping for a dynamically sized grid of static tiles.
///
/// Operands are invocation, component, rows, columns, and row stride. The
/// operation denotes the physical index only on the authenticated successful
/// path of the corresponding checked source operation.
#[pliron_op(
    name = "kernel.checked_tiled_index_2d",
    format,
    interfaces = [NRegionsInterface<0>],
    attributes = (
        kernel_lanes_per_tile: IndexValueAttr,
        kernel_tile_rows: IndexValueAttr,
        kernel_tile_columns: IndexValueAttr,
        kernel_elements_per_lane: IndexValueAttr
    )
)]
pub struct CheckedTiledIndex2DOp;

impl CheckedTiledIndex2DOp {
    pub fn new(
        context: &mut Context,
        invocation: Value,
        component: Value,
        rows: Value,
        columns: Value,
        row_stride: Value,
        geometry: [u64; 4],
    ) -> Self {
        let [lanes_per_tile, tile_rows, tile_columns, elements_per_lane] = geometry;
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![invocation, component, rows, columns, row_stride],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_lanes_per_tile(context, IndexValueAttr(lanes_per_tile));
        op.set_attr_kernel_tile_rows(context, IndexValueAttr(tile_rows));
        op.set_attr_kernel_tile_columns(context, IndexValueAttr(tile_columns));
        op.set_attr_kernel_elements_per_lane(context, IndexValueAttr(elements_per_lane));
        op
    }

    /// Builds the predicated structural form. The final operand is the
    /// physical extent of the one-dimensional destination view and result 1
    /// carries the obligation consumed by the corresponding access. This
    /// shape grants no source or refinement authority.
    #[allow(clippy::too_many_arguments)]
    pub fn new_predicated(
        context: &mut Context,
        invocation: Value,
        component: Value,
        rows: Value,
        columns: Value,
        row_stride: Value,
        physical_extent: Value,
        geometry: [u64; 4],
    ) -> Self {
        let [lanes_per_tile, tile_rows, tile_columns, elements_per_lane] = geometry;
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![
                IndexType::get(context).into(),
                CheckedAccessCapabilityType::get(context).into(),
            ],
            vec![
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_lanes_per_tile(context, IndexValueAttr(lanes_per_tile));
        op.set_attr_kernel_tile_rows(context, IndexValueAttr(tile_rows));
        op.set_attr_kernel_tile_columns(context, IndexValueAttr(tile_columns));
        op.set_attr_kernel_elements_per_lane(context, IndexValueAttr(elements_per_lane));
        op
    }

    pub fn operands(&self, context: &Context) -> [Value; 5] {
        let operation = self.get_operation().deref(context);
        core::array::from_fn(|index| operation.get_operand(index))
    }

    pub fn geometry(&self, context: &Context) -> Option<[u64; 4]> {
        Some([
            self.get_attr_kernel_lanes_per_tile(context)?.value(),
            self.get_attr_kernel_tile_rows(context)?.value(),
            self.get_attr_kernel_tile_columns(context)?.value(),
            self.get_attr_kernel_elements_per_lane(context)?.value(),
        ])
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }

    pub fn physical_extent(&self, context: &Context) -> Option<Value> {
        let operation = self.get_operation().deref(context);
        (operation.get_num_operands() == 6).then(|| operation.get_operand(5))
    }

    pub fn success(&self, context: &Context) -> Option<Value> {
        let operation = self.get_operation().deref(context);
        (operation.get_num_results() == 2).then(|| operation.get_result(1))
    }
}

impl Verify for CheckedTiledIndex2DOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        let raw = self.get_operation();
        let raw = raw.deref(context);
        verify_no_regions_results_successors(self, context, raw.get_num_results(), 0)?;
        let geometry = self.geometry(context);
        let legacy = raw.get_num_operands() == 5 && raw.get_num_results() == 1;
        let predicated = raw.get_num_operands() == 6
            && raw.get_num_results() == 2
            && is_checked_access_capability(raw.get_result(1), context);
        if !(legacy || predicated)
            || payload_attribute_count(&raw) != 4
            || geometry.is_none_or(|[lanes, rows, columns, elements]| {
                lanes == 0
                    || rows == 0
                    || columns == 0
                    || elements == 0
                    || !lanes.is_multiple_of(columns)
                    || lanes.checked_mul(elements) != rows.checked_mul(columns)
                    || (lanes / columns).checked_mul(elements) != Some(rows)
            })
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.checked_tiled_index_2d has malformed geometry or payload"
                )
            );
        }
        for operand in 0..raw.get_num_operands() {
            require_index_operand(self, context, operand)?;
        }
        Ok(())
    }
}

/// Checked compact row-striped index mapping.
///
/// Operands are invocation, component, rows, columns, and row stride. Static
/// attributes are lanes per row and elements per lane.
#[pliron_op(
    name = "kernel.checked_row_striped_index_2d",
    format,
    interfaces = [NRegionsInterface<0>],
    attributes = (
        kernel_lanes_per_row: IndexValueAttr,
        kernel_row_striped_elements_per_lane: IndexValueAttr
    )
)]
pub struct CheckedRowStripedIndex2DOp;

impl CheckedRowStripedIndex2DOp {
    pub fn new(
        context: &mut Context,
        invocation: Value,
        component: Value,
        rows: Value,
        columns: Value,
        row_stride: Value,
        geometry: [u64; 2],
    ) -> Self {
        let [lanes_per_row, elements_per_lane] = geometry;
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![invocation, component, rows, columns, row_stride],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_lanes_per_row(context, IndexValueAttr(lanes_per_row));
        op.set_attr_kernel_row_striped_elements_per_lane(
            context,
            IndexValueAttr(elements_per_lane),
        );
        op
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_predicated(
        context: &mut Context,
        invocation: Value,
        component: Value,
        rows: Value,
        columns: Value,
        row_stride: Value,
        physical_extent: Value,
        geometry: [u64; 2],
    ) -> Self {
        let [lanes_per_row, elements_per_lane] = geometry;
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![
                IndexType::get(context).into(),
                CheckedAccessCapabilityType::get(context).into(),
            ],
            vec![
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_lanes_per_row(context, IndexValueAttr(lanes_per_row));
        op.set_attr_kernel_row_striped_elements_per_lane(
            context,
            IndexValueAttr(elements_per_lane),
        );
        op
    }

    pub fn operands(&self, context: &Context) -> [Value; 5] {
        let operation = self.get_operation().deref(context);
        core::array::from_fn(|index| operation.get_operand(index))
    }

    pub fn geometry(&self, context: &Context) -> Option<[u64; 2]> {
        Some([
            self.get_attr_kernel_lanes_per_row(context)?.value(),
            self.get_attr_kernel_row_striped_elements_per_lane(context)?
                .value(),
        ])
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }

    pub fn physical_extent(&self, context: &Context) -> Option<Value> {
        let operation = self.get_operation().deref(context);
        (operation.get_num_operands() == 6).then(|| operation.get_operand(5))
    }

    pub fn success(&self, context: &Context) -> Option<Value> {
        let operation = self.get_operation().deref(context);
        (operation.get_num_results() == 2).then(|| operation.get_result(1))
    }
}

impl Verify for CheckedRowStripedIndex2DOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        let raw = self.get_operation();
        let raw = raw.deref(context);
        verify_no_regions_results_successors(self, context, raw.get_num_results(), 0)?;
        let geometry = self.geometry(context);
        let legacy = raw.get_num_operands() == 5 && raw.get_num_results() == 1;
        let predicated = raw.get_num_operands() == 6
            && raw.get_num_results() == 2
            && is_checked_access_capability(raw.get_result(1), context);
        if !(legacy || predicated)
            || payload_attribute_count(&raw) != 2
            || geometry.is_none_or(|[lanes, elements]| {
                lanes == 0
                    || elements == 0
                    || (elements - 1)
                        .checked_mul(lanes)
                        .and_then(|base| base.checked_add(lanes - 1))
                        .is_none()
            })
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.checked_row_striped_index_2d has malformed geometry or payload"
                )
            );
        }
        for operand in 0..raw.get_num_operands() {
            require_index_operand(self, context, operand)?;
        }
        Ok(())
    }
}

/// Reads one logical dimension from a ranked view.
#[pliron_op(
    name = "kernel.dim",
    format,
    interfaces = [NResultsInterface<1>, NRegionsInterface<0>],
    attributes = (kernel_dimension: DimensionAttr)
)]
pub struct DimensionOp;

impl DimensionOp {
    pub fn new(
        context: &mut Context,
        view: Value,
        dimension: u32,
    ) -> Result<Self, RankedMemoryError> {
        let view_type =
            ranked_view_type(view, context).ok_or(RankedMemoryError::ForeignViewType)?;
        let rank = view_type.deref(context).rank();
        if usize::try_from(dimension)
            .ok()
            .is_none_or(|dimension| dimension >= rank)
        {
            return Err(RankedMemoryError::DimensionOutOfBounds { dimension, rank });
        }
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![IndexType::get(context).into()],
            vec![view],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_dimension(context, DimensionAttr(dimension));
        Ok(op)
    }

    pub fn dimension(&self, context: &Context) -> Option<u32> {
        self.get_attr_kernel_dimension(context)
            .map(|dimension| dimension.dimension())
    }

    pub fn view(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn result(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_result(0)
    }
}

impl Verify for DimensionOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 1, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        if operation.get_num_operands() != 1
            || payload_attribute_count(&operation) != 1
            || !is_index_type(self.result(context), context)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.dim has malformed payload")
            );
        }
        let Some(view_type) = ranked_view_type(self.view(context), context) else {
            return verify_err!(self.loc(context), RankedMemoryError::ForeignViewType);
        };
        let rank = view_type.deref(context).rank();
        let Some(dimension) = self.dimension(context) else {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.dim is missing its dimension")
            );
        };
        if usize::try_from(dimension)
            .ok()
            .is_none_or(|dimension| dimension >= rank)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::DimensionOutOfBounds { dimension, rank }
            );
        }
        Ok(())
    }
}

/// Target-neutral indexed memory access. Operand 0 is the view; the rest are indices.
#[pliron_op(
    name = "kernel.access",
    format,
    interfaces = [NResultsInterface<0>, NRegionsInterface<0>],
    attributes = (
        kernel_access_kind: AccessKindAttr,
        kernel_atomic_ordering: AtomicOrderingAttr,
        kernel_atomic_scope: AtomicScopeAttr
    )
)]
pub struct RankedAccessOp;

impl RankedAccessOp {
    pub fn new(
        context: &mut Context,
        kind: AccessKindAttr,
        view: Value,
        indices: Vec<Value>,
    ) -> Result<Self, RankedMemoryError> {
        if kind.is_atomic() {
            return Err(RankedMemoryError::MissingAtomicContract);
        }
        Self::build(context, kind, None, None, view, indices, None)
    }

    pub fn new_atomic(
        context: &mut Context,
        kind: AccessKindAttr,
        ordering: AtomicOrderingAttr,
        scope: AtomicScopeAttr,
        view: Value,
        indices: Vec<Value>,
    ) -> Result<Self, RankedMemoryError> {
        if !kind.is_atomic() {
            return Err(RankedMemoryError::UnexpectedAtomicContract);
        }
        Self::build(
            context,
            kind,
            Some(ordering),
            Some(scope),
            view,
            indices,
            None,
        )
    }

    /// Builds a structurally predicated one-dimensional non-atomic access. The opaque
    /// success value is paired with the checked index and physical extent.
    /// This constructor grants no source or refinement authority.
    pub fn new_predicated(
        context: &mut Context,
        kind: AccessKindAttr,
        view: Value,
        index: Value,
        success: Value,
    ) -> Result<Self, RankedMemoryError> {
        if kind.is_atomic() {
            return Err(RankedMemoryError::MissingAtomicContract);
        }
        validate_predicated_access(context, view, index, success)?;
        Self::build(context, kind, None, None, view, vec![index], Some(success))
    }

    fn build(
        context: &mut Context,
        kind: AccessKindAttr,
        ordering: Option<AtomicOrderingAttr>,
        scope: Option<AtomicScopeAttr>,
        view: Value,
        indices: Vec<Value>,
        success: Option<Value>,
    ) -> Result<Self, RankedMemoryError> {
        let view_type =
            ranked_view_type(view, context).ok_or(RankedMemoryError::ForeignViewType)?;
        let (rank, writable) = {
            let view_type = view_type.deref(context);
            (view_type.rank(), view_type.writable())
        };
        if indices.len() != rank {
            return Err(RankedMemoryError::OperandCountMismatch {
                expected: rank,
                actual: indices.len(),
            });
        }
        if kind.writes_memory() && !writable {
            return Err(RankedMemoryError::WriteThroughReadOnlyView);
        }
        let mut operands = Vec::with_capacity(indices.len() + 1 + usize::from(success.is_some()));
        operands.push(view);
        operands.extend(indices);
        operands.extend(success);
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_access_kind(context, kind);
        if let Some(ordering) = ordering {
            op.set_attr_kernel_atomic_ordering(context, ordering);
        }
        if let Some(scope) = scope {
            op.set_attr_kernel_atomic_scope(context, scope);
        }
        Ok(op)
    }

    pub fn kind(&self, context: &Context) -> Option<AccessKindAttr> {
        self.get_attr_kernel_access_kind(context).map(|kind| *kind)
    }

    pub fn view(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn atomic_ordering(&self, context: &Context) -> Option<AtomicOrderingAttr> {
        self.get_attr_kernel_atomic_ordering(context)
            .map(|ordering| *ordering)
    }

    pub fn atomic_scope(&self, context: &Context) -> Option<AtomicScopeAttr> {
        self.get_attr_kernel_atomic_scope(context)
            .map(|scope| *scope)
    }

    pub fn indices(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let end = operation
            .get_num_operands()
            .saturating_sub(usize::from(self.checked_success(context).is_some()));
        (1..end)
            .map(|operand| operation.get_operand(operand))
            .collect()
    }

    pub fn checked_success(&self, context: &Context) -> Option<Value> {
        let operation = self.get_operation().deref(context);
        let last = operation.get_num_operands().checked_sub(1)?;
        let value = operation.get_operand(last);
        is_checked_access_capability(value, context).then_some(value)
    }
}

impl Verify for RankedAccessOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        if operation.get_num_operands() == 0
            || !(1..=3).contains(&payload_attribute_count(&operation))
            || self.kind(context).is_none()
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.access has malformed payload")
            );
        }
        let Some(view_type) = ranked_view_type(self.view(context), context) else {
            return verify_err!(self.loc(context), RankedMemoryError::ForeignViewType);
        };
        let view_type = view_type.deref(context);
        let success = self.checked_success(context);
        let actual = operation.get_num_operands() - 1 - usize::from(success.is_some());
        if actual != view_type.rank() {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch {
                    expected: view_type.rank(),
                    actual,
                }
            );
        }
        if self
            .kind(context)
            .is_some_and(AccessKindAttr::writes_memory)
            && !view_type.writable()
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::WriteThroughReadOnlyView
            );
        }
        for operand in 1..=actual {
            require_index_operand(self, context, operand)?;
        }
        if let Some(success) = success {
            if self.kind(context).is_none_or(AccessKindAttr::is_atomic) || actual != 1 {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "predicated kernel.access must be one non-atomic read or write"
                    )
                );
            }
            if let Err(error) = validate_predicated_access(
                context,
                self.view(context),
                self.indices(context)[0],
                success,
            ) {
                return verify_err!(self.loc(context), error);
            }
        }
        Ok(())
    }
}

fn validate_predicated_access(
    context: &Context,
    view: Value,
    index: Value,
    success: Value,
) -> Result<(), RankedMemoryError> {
    if !is_checked_access_capability(success, context) {
        return Err(RankedMemoryError::MalformedPayload(
            "predicated kernel.access success has the wrong type",
        ));
    }
    let view_type = ranked_view_type(view, context).ok_or(RankedMemoryError::ForeignViewType)?;
    if view_type.deref(context).shape() != [DYNAMIC_EXTENT] {
        return Err(RankedMemoryError::MalformedPayload(
            "predicated kernel.access requires one dynamic view extent",
        ));
    }
    let view_definition = view
        .defining_op()
        .ok_or(RankedMemoryError::ForeignViewType)?;
    let view_operation = Operation::get_op_dyn(view_definition, context);
    let view_operation = view_operation
        .downcast_ref::<RankedViewOp>()
        .ok_or(RankedMemoryError::ForeignViewType)?;
    let extent =
        view_operation
            .dynamic_extent(context, 0)
            .ok_or(RankedMemoryError::MalformedPayload(
                "predicated kernel.access view is missing its physical extent",
            ))?;
    let producer = success
        .defining_op()
        .ok_or(RankedMemoryError::MalformedPayload(
            "predicated kernel.access success has no checked definition",
        ))?;
    if index.defining_op() != Some(producer) {
        return Err(RankedMemoryError::MalformedPayload(
            "predicated kernel.access index and success have different definitions",
        ));
    }
    let producer = Operation::get_op_dyn(producer, context);
    let (expected_index, expected_success, expected_extent) =
        if let Some(checked) = producer.downcast_ref::<CheckedTiledIndex2DOp>() {
            (
                checked.result(context),
                checked.success(context),
                checked.physical_extent(context),
            )
        } else if let Some(checked) = producer.downcast_ref::<CheckedRowStripedIndex2DOp>() {
            (
                checked.result(context),
                checked.success(context),
                checked.physical_extent(context),
            )
        } else {
            return Err(RankedMemoryError::MalformedPayload(
                "predicated kernel.access success has a foreign definition",
            ));
        };
    if expected_index != index
        || expected_success != Some(success)
        || expected_extent != Some(extent)
    {
        return Err(RankedMemoryError::MalformedPayload(
            "predicated kernel.access changed its checked index, success, or physical extent",
        ));
    }
    Ok(())
}

/// Requests a hierarchy-level ownership proof for one logical output view.
///
/// This operation is inert metadata. Its local verifier establishes only that
/// the operand is a writable global ranked view and that the payload is
/// closed. The hierarchy ownership compiler pass proves bounds, injectivity,
/// coverage, and subgroup/workgroup/grid partitioning from actual writes.
#[pliron_op(
    name = "kernel.ownership_contract",
    format,
    interfaces = [NResultsInterface<0>, NRegionsInterface<0>],
    attributes = (
        kernel_ownership_coverage: OwnershipCoverageAttr,
        kernel_ownership_partition: OwnershipPartitionAttr
    )
)]
pub struct OwnershipContractOp;

impl OwnershipContractOp {
    pub fn new(
        context: &mut Context,
        view: Value,
        coverage: OwnershipCoverageAttr,
        partition: OwnershipPartitionAttr,
    ) -> Result<Self, RankedMemoryError> {
        let view_type =
            ranked_view_type(view, context).ok_or(RankedMemoryError::ForeignViewType)?;
        if !view_type.deref(context).writable() {
            return Err(RankedMemoryError::WriteThroughReadOnlyView);
        }
        let Some(definition) = view.defining_op() else {
            return Err(RankedMemoryError::ForeignViewType);
        };
        let definition = Operation::get_op_dyn(definition, context);
        let Some(view_op) = definition.downcast_ref::<RankedViewOp>() else {
            return Err(RankedMemoryError::ForeignViewType);
        };
        if view_op.memory_space(context) != Some(MemorySpaceAttr::Global) {
            return Err(RankedMemoryError::MalformedPayload(
                "kernel.ownership_contract requires a global ranked view",
            ));
        }
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![view],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_ownership_coverage(context, coverage);
        op.set_attr_kernel_ownership_partition(context, partition);
        Ok(op)
    }

    pub fn view(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn coverage(&self, context: &Context) -> Option<OwnershipCoverageAttr> {
        self.get_attr_kernel_ownership_coverage(context)
            .map(|coverage| *coverage)
    }

    pub fn partition(&self, context: &Context) -> Option<OwnershipPartitionAttr> {
        self.get_attr_kernel_ownership_partition(context)
            .map(|partition| *partition)
    }
}

impl Verify for OwnershipContractOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        if operation.get_num_operands() != 1
            || payload_attribute_count(&operation) != 2
            || self.coverage(context).is_none()
            || self.partition(context).is_none()
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.ownership_contract has malformed payload"
                )
            );
        }
        let view = self.view(context);
        let Some(view_type) = ranked_view_type(view, context) else {
            return verify_err!(self.loc(context), RankedMemoryError::ForeignViewType);
        };
        if !view_type.deref(context).writable() {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::WriteThroughReadOnlyView
            );
        }
        let Some(definition) = view.defining_op() else {
            return verify_err!(self.loc(context), RankedMemoryError::ForeignViewType);
        };
        let definition = Operation::get_op_dyn(definition, context);
        if definition
            .downcast_ref::<RankedViewOp>()
            .and_then(|view| view.memory_space(context))
            != Some(MemorySpaceAttr::Global)
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.ownership_contract requires a global ranked view"
                )
            );
        }
        Ok(())
    }
}

/// Conservative whole-allocation memory effect with no fabricated coordinate.
#[pliron_op(
    name = "kernel.allocation_effect",
    format,
    interfaces = [NResultsInterface<0>, NRegionsInterface<0>],
    attributes = (
        kernel_allocation_effect_access_kind: AccessKindAttr,
        kernel_allocation_effect_memory_space: MemorySpaceAttr,
        kernel_allocation_effect_origin: AllocationOriginAttr,
        kernel_allocation_effect_noalias_class: NoAliasClassAttr
    )
)]
pub struct AllocationEffectOp;

impl AllocationEffectOp {
    pub fn new(
        context: &mut Context,
        kind: AccessKindAttr,
        memory_space: MemorySpaceAttr,
        allocation_origin: u64,
        noalias_class: u64,
    ) -> Result<Self, RankedMemoryError> {
        if !is_supported_allocation_effect_contract_v1(
            kind,
            memory_space,
            allocation_origin,
            noalias_class,
        ) {
            return Err(RankedMemoryError::MalformedPayload(
                "kernel.allocation_effect has an unsupported allocation contract",
            ));
        }
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let op = Self::from_operation(operation);
        op.set_attr_kernel_allocation_effect_access_kind(context, kind);
        op.set_attr_kernel_allocation_effect_memory_space(context, memory_space);
        op.set_attr_kernel_allocation_effect_origin(
            context,
            AllocationOriginAttr(allocation_origin),
        );
        op.set_attr_kernel_allocation_effect_noalias_class(
            context,
            NoAliasClassAttr(noalias_class),
        );
        Ok(op)
    }

    pub fn kind(&self, context: &Context) -> Option<AccessKindAttr> {
        self.get_attr_kernel_allocation_effect_access_kind(context)
            .map(|kind| *kind)
    }

    pub fn memory_space(&self, context: &Context) -> Option<MemorySpaceAttr> {
        self.get_attr_kernel_allocation_effect_memory_space(context)
            .map(|space| *space)
    }

    pub fn allocation_origin(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_allocation_effect_origin(context)
            .map(|origin| origin.identity())
    }

    pub fn noalias_class(&self, context: &Context) -> Option<u64> {
        self.get_attr_kernel_allocation_effect_noalias_class(context)
            .map(|class| class.identity())
    }
}

impl Verify for AllocationEffectOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 0)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let valid_contract = match (
            self.kind(context),
            self.memory_space(context),
            self.allocation_origin(context),
            self.noalias_class(context),
        ) {
            (Some(kind), Some(memory_space), Some(allocation_origin), Some(noalias_class)) => {
                is_supported_allocation_effect_contract_v1(
                    kind,
                    memory_space,
                    allocation_origin,
                    noalias_class,
                )
            }
            _ => false,
        };
        if operation.get_num_operands() != 0
            || payload_attribute_count(&operation) != 4
            || !valid_contract
        {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.allocation_effect has malformed payload"
                )
            );
        }
        Ok(())
    }
}

/// Conditional branch whose first successor is selected exactly when `lhs < rhs`.
#[pliron_op(
    name = "kernel.index_lt_br",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct IndexLessThanBranchOp;

impl IndexLessThanBranchOp {
    pub fn new(
        context: &mut Context,
        lhs: Value,
        rhs: Value,
        true_successor: Ptr<pliron::basic_block::BasicBlock>,
        false_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![lhs, rhs],
            vec![true_successor, false_successor],
            0,
        ))
    }

    pub fn lhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn rhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(1)
    }
}

impl Verify for IndexLessThanBranchOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 2)?;
        if self.get_operation().deref(context).get_num_operands() != 2 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch {
                    expected: 2,
                    actual: self.get_operation().deref(context).get_num_operands(),
                }
            );
        }
        require_index_operand(self, context, 0)?;
        require_index_operand(self, context, 1)?;
        require_zero_successor_arguments(self, context)
    }
}

/// Index comparison branch carrying exact SSA values to both successors.
#[pliron_op(
    name = "kernel.index_lt_br_args",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct IndexLessThanBranchArgsOp;

impl IndexLessThanBranchArgsOp {
    pub fn new(
        context: &mut Context,
        lhs: Value,
        rhs: Value,
        true_arguments: Vec<Value>,
        false_arguments: Vec<Value>,
        true_successor: Ptr<pliron::basic_block::BasicBlock>,
        false_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        let mut operands = Vec::with_capacity(2 + true_arguments.len() + false_arguments.len());
        operands.push(lhs);
        operands.push(rhs);
        operands.extend(true_arguments);
        operands.extend(false_arguments);
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_successor, false_successor],
            0,
        ))
    }

    pub fn lhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn rhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(1)
    }

    pub fn true_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        (0..count)
            .map(|index| operation.get_operand(2 + index))
            .collect()
    }

    pub fn false_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let true_count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        let false_count = operation
            .get_successor(1)
            .deref(context)
            .get_num_arguments();
        (0..false_count)
            .map(|index| operation.get_operand(2 + true_count + index))
            .collect()
    }
}

impl Verify for IndexLessThanBranchArgsOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 2)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let true_successor = operation.get_successor(0);
        let false_successor = operation.get_successor(1);
        let true_count = true_successor.deref(context).get_num_arguments();
        let false_count = false_successor.deref(context).get_num_arguments();
        let expected = 2 + true_count + false_count;
        let actual = operation.get_num_operands();
        if actual != expected {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch { expected, actual }
            );
        }
        require_index_operand(self, context, 0)?;
        require_index_operand(self, context, 1)?;
        for index in 0..true_count {
            if operation.get_operand(2 + index).get_type(context)
                != true_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.index_lt_br_args true-edge types differ",
                    )
                );
            }
        }
        for index in 0..false_count {
            if operation
                .get_operand(2 + true_count + index)
                .get_type(context)
                != false_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.index_lt_br_args false-edge types differ",
                    )
                );
            }
        }
        Ok(())
    }
}

/// Conditional branch whose first successor is selected exactly when `lhs == rhs`.
#[pliron_op(
    name = "kernel.index_eq_br",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct IndexEqualBranchOp;

impl IndexEqualBranchOp {
    pub fn new(
        context: &mut Context,
        lhs: Value,
        rhs: Value,
        true_successor: Ptr<pliron::basic_block::BasicBlock>,
        false_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![lhs, rhs],
            vec![true_successor, false_successor],
            0,
        ))
    }

    pub fn lhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn rhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(1)
    }
}

impl Verify for IndexEqualBranchOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 2)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        if operation.get_num_operands() != 2 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch {
                    expected: 2,
                    actual: operation.get_num_operands(),
                }
            );
        }
        if payload_attribute_count(&operation) != 0 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.index_eq_br carries unexpected attributes",
                )
            );
        }
        require_index_operand(self, context, 0)?;
        require_index_operand(self, context, 1)?;
        require_zero_successor_arguments(self, context)
    }
}

/// Equality branch carrying exact SSA values to both successors.
#[pliron_op(
    name = "kernel.index_eq_br_args",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct IndexEqualBranchArgsOp;

impl IndexEqualBranchArgsOp {
    pub fn new(
        context: &mut Context,
        lhs: Value,
        rhs: Value,
        true_arguments: Vec<Value>,
        false_arguments: Vec<Value>,
        true_successor: Ptr<pliron::basic_block::BasicBlock>,
        false_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        let mut operands = Vec::with_capacity(2 + true_arguments.len() + false_arguments.len());
        operands.push(lhs);
        operands.push(rhs);
        operands.extend(true_arguments);
        operands.extend(false_arguments);
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_successor, false_successor],
            0,
        ))
    }

    pub fn lhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(0)
    }

    pub fn rhs(&self, context: &Context) -> Value {
        self.get_operation().deref(context).get_operand(1)
    }

    pub fn true_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        (0..count)
            .map(|index| operation.get_operand(2 + index))
            .collect()
    }

    pub fn false_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let true_count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        let false_count = operation
            .get_successor(1)
            .deref(context)
            .get_num_arguments();
        (0..false_count)
            .map(|index| operation.get_operand(2 + true_count + index))
            .collect()
    }
}

impl Verify for IndexEqualBranchArgsOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 2)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let true_successor = operation.get_successor(0);
        let false_successor = operation.get_successor(1);
        let true_count = true_successor.deref(context).get_num_arguments();
        let false_count = false_successor.deref(context).get_num_arguments();
        let expected = 2 + true_count + false_count;
        let actual = operation.get_num_operands();
        if actual != expected {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch { expected, actual }
            );
        }
        if payload_attribute_count(&operation) != 0 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.index_eq_br_args carries unexpected attributes",
                )
            );
        }
        require_index_operand(self, context, 0)?;
        require_index_operand(self, context, 1)?;
        for index in 0..true_count {
            if operation.get_operand(2 + index).get_type(context)
                != true_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.index_eq_br_args true-edge types differ",
                    )
                );
            }
        }
        for index in 0..false_count {
            if operation
                .get_operand(2 + true_count + index)
                .get_type(context)
                != false_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.index_eq_br_args false-edge types differ",
                    )
                );
            }
        }
        Ok(())
    }
}

/// Target-neutral two-way split that retains the complete control-dependency
/// set without interpreting the source predicate.
#[pliron_op(
    name = "kernel.analysis_split",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>],
    attributes = (kernel_analysis_split_control_count: AnalysisSplitControlCountAttr)
)]
pub struct AnalysisSplitOp;

impl AnalysisSplitOp {
    pub fn new(
        context: &mut Context,
        first_successor: Ptr<pliron::basic_block::BasicBlock>,
        second_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        Self::new_with_control_and_arguments(
            context,
            vec![],
            vec![],
            vec![],
            first_successor,
            second_successor,
        )
    }

    pub fn new_with_arguments(
        context: &mut Context,
        first_arguments: Vec<Value>,
        second_arguments: Vec<Value>,
        first_successor: Ptr<pliron::basic_block::BasicBlock>,
        second_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        Self::new_with_control_and_arguments(
            context,
            vec![],
            first_arguments,
            second_arguments,
            first_successor,
            second_successor,
        )
    }

    pub fn new_with_control_and_arguments(
        context: &mut Context,
        control_dependencies: Vec<Value>,
        first_arguments: Vec<Value>,
        second_arguments: Vec<Value>,
        first_successor: Ptr<pliron::basic_block::BasicBlock>,
        second_successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        let control_count = u32::try_from(control_dependencies.len()).unwrap_or(u32::MAX);
        let mut operands = Vec::with_capacity(
            control_dependencies.len() + first_arguments.len() + second_arguments.len(),
        );
        operands.extend(control_dependencies);
        operands.extend(first_arguments);
        operands.extend(second_arguments);
        let op = Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![first_successor, second_successor],
            0,
        ));
        op.set_attr_kernel_analysis_split_control_count(
            context,
            AnalysisSplitControlCountAttr(control_count),
        );
        op
    }

    pub fn control_dependencies(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let count = self
            .get_attr_kernel_analysis_split_control_count(context)
            .map_or(0, |count| count.count() as usize);
        (0..count.min(operation.get_num_operands()))
            .map(|index| operation.get_operand(index))
            .collect()
    }

    pub fn first_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        let control_count = self
            .get_attr_kernel_analysis_split_control_count(context)
            .map_or(0, |count| count.count() as usize);
        (0..count)
            .map(|index| operation.get_operand(control_count + index))
            .collect()
    }

    pub fn second_arguments(&self, context: &Context) -> Vec<Value> {
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let first_count = operation
            .get_successor(0)
            .deref(context)
            .get_num_arguments();
        let second_count = operation
            .get_successor(1)
            .deref(context)
            .get_num_arguments();
        let control_count = self
            .get_attr_kernel_analysis_split_control_count(context)
            .map_or(0, |count| count.count() as usize);
        (0..second_count)
            .map(|index| operation.get_operand(control_count + first_count + index))
            .collect()
    }
}

impl Verify for AnalysisSplitOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 2)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let first_successor = operation.get_successor(0);
        let second_successor = operation.get_successor(1);
        let first_count = first_successor.deref(context).get_num_arguments();
        let second_count = second_successor.deref(context).get_num_arguments();
        let Some(control_count) = self
            .get_attr_kernel_analysis_split_control_count(context)
            .and_then(|count| usize::try_from(count.count()).ok())
        else {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.analysis_split requires one valid control-count segment",
                )
            );
        };
        let expected = control_count
            .checked_add(first_count)
            .and_then(|count| count.checked_add(second_count))
            .unwrap_or(usize::MAX);
        let actual = operation.get_num_operands();
        if actual != expected {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch { expected, actual }
            );
        }
        if payload_attribute_count(&operation) != 1 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload(
                    "kernel.analysis_split carries unexpected attributes",
                )
            );
        }
        for index in 0..control_count {
            if operation.get_operand(index).get_type(context) != IndexType::get(context).into() {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.analysis_split control dependency is not a kernel index",
                    )
                );
            }
        }
        for index in 0..first_count {
            if operation
                .get_operand(control_count + index)
                .get_type(context)
                != first_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.analysis_split first-edge types differ",
                    )
                );
            }
        }
        for index in 0..second_count {
            if operation
                .get_operand(control_count + first_count + index)
                .get_type(context)
                != second_successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.analysis_split second-edge types differ",
                    )
                );
            }
        }
        Ok(())
    }
}

/// Unconditional target-neutral branch.
#[pliron_op(
    name = "kernel.br",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct BranchOp;

impl BranchOp {
    pub fn new(context: &mut Context, successor: Ptr<pliron::basic_block::BasicBlock>) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![successor],
            0,
        ))
    }
}

impl Verify for BranchOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 1)?;
        if self.get_operation().deref(context).get_num_operands() != 0 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.br cannot carry operands")
            );
        }
        require_zero_successor_arguments(self, context)
    }
}

/// Unconditional branch carrying exact SSA values to successor block arguments.
#[pliron_op(
    name = "kernel.br_args",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct BranchArgsOp;

impl BranchArgsOp {
    pub fn new(
        context: &mut Context,
        arguments: Vec<Value>,
        successor: Ptr<pliron::basic_block::BasicBlock>,
    ) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            arguments,
            vec![successor],
            0,
        ))
    }

    pub fn arguments(&self, context: &Context) -> Vec<Value> {
        self.get_operation().deref(context).operands().collect()
    }
}

impl Verify for BranchArgsOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 1)?;
        let operation = self.get_operation();
        let operation = operation.deref(context);
        let successor = operation.get_successor(0);
        let expected = successor.deref(context).get_num_arguments();
        let actual = operation.get_num_operands();
        if actual != expected {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::OperandCountMismatch { expected, actual }
            );
        }
        for index in 0..actual {
            if operation.get_operand(index).get_type(context)
                != successor
                    .deref(context)
                    .get_argument(index)
                    .get_type(context)
            {
                return verify_err!(
                    self.loc(context),
                    RankedMemoryError::MalformedPayload(
                        "kernel.br_args operand and successor argument types differ",
                    )
                );
            }
        }
        Ok(())
    }
}

/// Verifies the shared parent and signature contract for kernel terminators.
fn verify_void_function_terminator<O: Op>(
    terminator: &O,
    context: &Context,
    parent_message: &'static str,
    foreign_type_message: &'static str,
    signature_message: &'static str,
) -> PlironResult<()> {
    let Some(parent) = terminator
        .get_operation()
        .deref(context)
        .get_parent_op(context)
    else {
        return verify_err!(
            terminator.loc(context),
            RankedMemoryError::MalformedPayload(parent_message)
        );
    };
    if !Operation::is_op::<FuncOp>(parent, context) {
        return verify_err!(
            terminator.loc(context),
            RankedMemoryError::MalformedPayload(parent_message)
        );
    }
    let function = FuncOp::from_operation(parent);
    let Ok(function_type) =
        TypedHandle::<FunctionType>::from_handle(function.get_type(context), context)
    else {
        return verify_err!(
            terminator.loc(context),
            RankedMemoryError::MalformedPayload(foreign_type_message)
        );
    };
    if !function_type.deref(context).res_types().is_empty() {
        return verify_err!(
            terminator.loc(context),
            RankedMemoryError::MalformedPayload(signature_message)
        );
    }
    Ok(())
}

/// Terminates a void target-neutral kernel function successfully.
#[pliron_op(
    name = "kernel.return",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(context: &mut Context) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        ))
    }
}

impl Verify for ReturnOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 0)?;
        if self.get_operation().deref(context).get_num_operands() != 0 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.return cannot carry operands")
            );
        }
        verify_void_function_terminator(
            self,
            context,
            "kernel.return must directly terminate a builtin.func block",
            "kernel.return parent has a foreign type",
            "kernel.return requires a void builtin.func signature",
        )
    }
}

/// Terminates a target-neutral kernel function by trapping the current invocation.
#[pliron_op(
    name = "kernel.trap",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NRegionsInterface<0>]
)]
pub struct TrapOp;

impl TrapOp {
    pub fn new(context: &mut Context) -> Self {
        Self::from_operation(Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        ))
    }
}

impl Verify for TrapOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_no_regions_results_successors(self, context, 0, 0)?;
        if self.get_operation().deref(context).get_num_operands() != 0 {
            return verify_err!(
                self.loc(context),
                RankedMemoryError::MalformedPayload("kernel.trap cannot carry operands")
            );
        }
        verify_void_function_terminator(
            self,
            context,
            "kernel.trap must directly terminate a builtin.func block",
            "kernel.trap parent has a foreign type",
            "kernel.trap requires a void builtin.func signature",
        )
    }
}

pub fn ranked_view_type(value: Value, context: &Context) -> Option<TypedHandle<RankedViewType>> {
    TypedHandle::from_handle(value.get_type(context), context).ok()
}

pub fn is_index_type(value: Value, context: &Context) -> bool {
    value.get_type(context).deref(context).is::<IndexType>()
}

fn is_checked_access_capability(value: Value, context: &Context) -> bool {
    is_checked_access_capability_type(&*value.get_type(context).deref(context))
}

/// Recognizes the opaque structural carrier without exposing its constructor.
pub fn is_checked_access_capability_type(ty: &dyn Type) -> bool {
    ty.downcast_ref::<CheckedAccessCapabilityType>().is_some()
}

fn require_index_operand(
    operation: &dyn Op,
    context: &Context,
    operand: usize,
) -> PlironResult<()> {
    let value = operation
        .get_operation()
        .deref(context)
        .get_operand(operand);
    if !is_index_type(value, context) {
        return verify_err!(
            operation.loc(context),
            RankedMemoryError::ForeignIndexType { operand }
        );
    }
    Ok(())
}

fn require_zero_successor_arguments(operation: &dyn Op, context: &Context) -> PlironResult<()> {
    let raw = operation.get_operation();
    let raw = raw.deref(context);
    if raw
        .successors()
        .any(|successor| successor.deref(context).get_num_arguments() != 0)
    {
        return verify_err!(
            operation.loc(context),
            RankedMemoryError::MalformedPayload("branch omits required successor block arguments",)
        );
    }
    Ok(())
}

fn verify_no_regions_results_successors(
    operation: &dyn Op,
    context: &Context,
    results: usize,
    successors: usize,
) -> PlironResult<()> {
    let raw = operation.get_operation();
    let raw = raw.deref(context);
    let attributes_are_closed = raw.attributes.0.keys().all(|key| {
        key == &*ATTR_KEY_DEBUG_INFO
            || matches!(
                key.as_ref(),
                "kernel_index_value"
                    | "kernel_unsigned_bit_width"
                    | "kernel_dimension"
                    | "kernel_access_kind"
                    | "kernel_atomic_ordering"
                    | "kernel_atomic_scope"
                    | "kernel_memory_space"
                    | "kernel_allocation_origin"
                    | "kernel_noalias_class"
                    | "kernel_allocation_effect_access_kind"
                    | "kernel_allocation_effect_memory_space"
                    | "kernel_allocation_effect_origin"
                    | "kernel_allocation_effect_noalias_class"
                    | "kernel_invocation_dimension"
                    | "kernel_launch_extent"
                    | "kernel_analysis_split_control_count"
                    | "kernel_index_binary_kind"
                    | "kernel_lanes_per_tile"
                    | "kernel_tile_rows"
                    | "kernel_tile_columns"
                    | "kernel_elements_per_lane"
                    | "kernel_lanes_per_row"
                    | "kernel_row_striped_elements_per_lane"
                    | "kernel_ownership_coverage"
                    | "kernel_ownership_partition"
            )
    });
    if raw.get_num_results() != results
        || raw.get_num_successors() != successors
        || raw.num_regions() != 0
        || !attributes_are_closed
    {
        return verify_err!(
            operation.loc(context),
            RankedMemoryError::MalformedPayload(
                "ranked-memory operation has malformed results, successors, regions, or attributes",
            )
        );
    }
    Ok(())
}

fn payload_attribute_count(operation: &Operation) -> usize {
    operation
        .attributes
        .0
        .keys()
        .filter(|key| *key != &*ATTR_KEY_DEBUG_INFO)
        .count()
}

pub const GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1: u64 = 0x5452_4e53_0000_0001;
pub const GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1: u64 = 0x5452_4e53_8000_0001;
pub const GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1: u64 = 0x5452_4e53_0000_0002;
pub const GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1: u64 = 0x5452_4e53_8000_0002;

pub const fn is_supported_allocation_effect_contract_v1(
    kind: AccessKindAttr,
    memory_space: MemorySpaceAttr,
    allocation_origin: u64,
    noalias_class: u64,
) -> bool {
    match (kind, memory_space, allocation_origin, noalias_class) {
        (AccessKindAttr::Read, MemorySpaceAttr::Global, origin, class) => class == 0 || origin != 0,
        (
            AccessKindAttr::Read | AccessKindAttr::Write,
            MemorySpaceAttr::Workgroup,
            GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        )
        | (
            AccessKindAttr::Read | AccessKindAttr::Write,
            MemorySpaceAttr::Workgroup,
            GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
            GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
        ) => true,
        _ => false,
    }
}
