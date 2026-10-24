//! Shared mutable checked-access emission. Component data, never admission.
//! The paid caller owns every partial vector outside all source/facts callbacks.
use super::bf16_nominal_preparation_resources_v1::{PreparationResourcesV1, resource};
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceErrorV1 as Resource, CanonicalKernelIrWorkLedgerIdentityV1,
};
type R<T> = Result<T, ProductionRankedProjectionErrorV1>;

#[derive(Debug)]
pub(super) struct AccessScratchV1 {
    pub(super) operation: Option<ProductionRankedOperationV1>,
    pub(super) comparisons: Vec<(ProductionRankedValueV1, ProductionRankedValueV1)>,
    pub(super) access: Option<GuardedRankedAccessV1>,
    pub(super) predicate: Option<GuardPredicateV1>,
}
impl AccessScratchV1 {
    pub(super) const fn empty() -> Self {
        Self {
            operation: None,
            comparisons: Vec::new(),
            access: None,
            predicate: None,
        }
    }
    fn clear(&self) -> bool {
        self.operation.is_none()
            && self.comparisons.is_empty()
            && self.comparisons.capacity() == 0
            && self.access.is_none()
            && self.predicate.is_none()
    }
}

/// Paid source-call identity retained alongside the exact appended guard vector.
/// Provenance/payload equality and cardinality alone are not identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RootGuardedSourceCallV1 {
    pub(super) source_call_ordinal: usize,
    pub(super) block: usize,
    pub(super) callee: SemanticCallableIdV1,
    pub(super) destination: SemanticLocalIdV1,
    pub(super) guarded_access: usize,
}

/// This extends the physical S3 assembly; it cannot construct an actual input loan.
pub(super) struct RootGuardedAccessStorageV1 {
    pub(super) views: Vec<Option<ProjectedViewV1>>,
    pub(super) accesses: Vec<GuardedRankedAccessV1>,
    pub(super) source_calls: Vec<RootGuardedSourceCallV1>,
    pub(super) scratch: AccessScratchV1,
    pub(super) ledger: Option<(usize, CanonicalKernelIrWorkLedgerIdentityV1)>,
    pub(super) started: bool,
    pub(super) completed: bool,
    pub(super) frame_credits: usize,
}
impl RootGuardedAccessStorageV1 {
    pub(super) const fn empty() -> Self {
        Self {
            views: Vec::new(),
            accesses: Vec::new(),
            source_calls: Vec::new(),
            scratch: AccessScratchV1::empty(),
            ledger: None,
            started: false,
            completed: false,
            frame_credits: 0,
        }
    }
    pub(super) fn completed(&self) -> bool {
        self.completed
    }
}

pub(super) fn identity_operand_v1(
    call: &SemanticDirectCallV1,
    index_values: &[Option<ProjectedDisjointIndexV1>],
    option_dominance: &SemanticOptionDominanceV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    block_index: usize,
) -> R<ProjectedDisjointIndexV1> {
    let projected = projected_disjoint_operand_v1(
        call,
        1,
        index_values,
        option_dominance,
        enum_payload_dominance,
        block_index,
    )?;
    if projected.mapping != SemanticDisjointIndexSpaceV1::Index1d {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "identity accessor received a non-identity mapping",
        ));
    }
    Ok(projected)
}

fn reserve_view_operation_v1(
    operations: &mut Vec<ProductionRankedOperationV1>,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> R<()> {
    if !resources.is_metered() {
        return reserve_operation(operations);
    }
    resources.work(16)?;
    if operations.len() == MAX_PROJECTED_OPERATIONS_V1 {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "a semantic intrinsic projection exceeding the ranked operation limit",
        ));
    }
    resources.reserve(operations, 1)
}

#[allow(clippy::too_many_arguments)]
fn new_view_v1(
    origin_index: usize,
    element_width: u32,
    allocation_contract: AllocationContractV1,
    views_by_origin: &mut [Option<ProjectedViewV1>],
    operations: &mut Vec<ProductionRankedOperationV1>,
    next_value: &mut u32,
    ranked_ir: Option<&mut String>,
    scratch: &mut AccessScratchV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> R<ProductionRankedValueIdV1> {
    reserve_view_operation_v1(operations, resources)?;
    resources.work(8)?;
    let view = next_value_id(next_value)?;
    if resources.is_metered() {
        // The shell is in the OUTER pending owner before either nested growth.
        scratch.operation = Some(ProductionRankedOperationV1::ViewInSpace {
            result: view,
            element_width,
            writable: true,
            shape: Vec::new(),
            dynamic_extents: Vec::new(),
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: allocation_contract.allocation_origin,
            noalias_class: allocation_contract.noalias_class,
        });
        let Some(ProductionRankedOperationV1::ViewInSpace {
            shape,
            dynamic_extents,
            ..
        }) = scratch.operation.as_mut()
        else {
            unreachable!()
        };
        resources.push(shape, DYNAMIC_EXTENT)?;
        resources.push(dynamic_extents, ProductionRankedValueV1::Argument(0))?;
        resources.work(1)?;
        operations.push(scratch.operation.take().expect("retained view operation"));
    } else {
        operations.push(ProductionRankedOperationV1::ViewInSpace {
            result: view,
            element_width,
            writable: true,
            shape: vec![DYNAMIC_EXTENT],
            dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: allocation_contract.allocation_origin,
            noalias_class: allocation_contract.noalias_class,
        });
    }
    // Paid preparation deliberately has no discarded diagnostic allocation.
    // Ordinary formatting is still AFTER push and BEFORE origin slot validation.
    if let Some(ranked_ir) = ranked_ir {
        push_ranked_ir(
            ranked_ir,
            &format!(
                "  %{} = kernel.ranked_view <{}, true, [dynamic], Global>(%arg0)\n",
                view.get(),
                element_width,
            ),
        )?;
    }
    let slot = views_by_origin.get_mut(origin_index).ok_or(
        ProductionRankedProjectionErrorV1::Unsupported(
            "a kernel argument origin outside the semantic local table",
        ),
    )?;
    if resources.is_metered() {
        *slot = Some(ProjectedViewV1 {
            result: view,
            element_width,
            writable: true,
            shape: Vec::new(),
            dynamic_extents: Vec::new(),
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: allocation_contract.allocation_origin,
            noalias_class: allocation_contract.noalias_class,
        });
        let stored = slot.as_mut().expect("outer cached view");
        resources.push(&mut stored.shape, DYNAMIC_EXTENT)?;
        resources.push(
            &mut stored.dynamic_extents,
            ProductionRankedValueV1::Argument(0),
        )?;
    } else {
        *slot = Some(ProjectedViewV1 {
            result: view,
            element_width,
            writable: true,
            shape: vec![DYNAMIC_EXTENT],
            dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: allocation_contract.allocation_origin,
            noalias_class: allocation_contract.noalias_class,
        });
    }
    Ok(view)
}

/// Raw shared emission helper. Only the joined factory supplies authentic source
/// data; inert component controls can call this without creating any authority.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_mutable_access_v1(
    types: &[SemanticTypeDeclV1],
    call: &SemanticDirectCallV1,
    block_index: usize,
    source: SemanticSourceProvenanceV1,
    element: SemanticTypeIdV1,
    index: ProductionRankedValueV1,
    precondition: Option<(ProductionRankedValueV1, ProductionRankedValueV1)>,
    checked_success: Option<ProductionRankedValueV1>,
    direct_write: bool,
    local_allocations: &[Option<AllocationContractV1>],
    allocation_provenance: &[Option<LocalAllocationProvenanceV1>],
    views_by_origin: &mut [Option<ProjectedViewV1>],
    guarded_accesses: &mut Vec<GuardedRankedAccessV1>,
    option_predicates: &mut [Option<GuardPredicateV1>],
    direct_write_effects: &mut [Option<GuardedRankedAccessV1>],
    operations: &mut Vec<ProductionRankedOperationV1>,
    next_value: &mut u32,
    ranked_ir: Option<&mut String>,
    scratch: &mut AccessScratchV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> R<()> {
    resources.work(256)?; // Fixed operand/allocation/type/destination queries.
    if resources.is_metered() && (direct_write || ranked_ir.is_some() || !scratch.clear()) {
        return Err(resource(Resource::Accounting));
    }
    if resources.is_metered() {
        if let Some(SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) =
            call.arguments().first()
        {
            resources.work(place.projections().len())?;
        }
    }
    let receiver = call
        .arguments()
        .first()
        .and_then(simple_operand_local)
        .ok_or(ProductionRankedProjectionErrorV1::Incomplete(
            "a checked disjoint receiver without one exact local",
        ))?
        .index() as usize;
    let allocation_contract = local_allocations.get(receiver).copied().flatten().ok_or(
        ProductionRankedProjectionErrorV1::Incomplete(
            "a checked disjoint receiver without one authenticated kernel-argument origin",
        ),
    )?;
    if !allocation_contract.writable {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "a checked mutable access is rooted in a read-only Rust allocation",
        ));
    }
    let origin_index = allocation_contract.allocation_origin as usize;
    let element_width = type_width(types, element)?;
    if resources.is_metered() {
        if let Some(view) = views_by_origin.get(origin_index).and_then(Option::as_ref) {
            resources.work(
                view.shape
                    .len()
                    .checked_add(view.dynamic_extents.len())
                    .and_then(|n| n.checked_add(32))
                    .ok_or_else(|| resource(Resource::Arithmetic))?,
            )?;
        }
    }
    let view = match views_by_origin
        .get(origin_index)
        .and_then(|view| view.as_ref())
    {
        Some(view)
            if view.element_width == element_width
                && view.writable
                && view.shape == [DYNAMIC_EXTENT]
                && view.dynamic_extents == [ProductionRankedValueV1::Argument(0)]
                && view.memory_space == MemorySpaceAttr::Global
                && view.allocation_origin == allocation_contract.allocation_origin
                && view.noalias_class == allocation_contract.noalias_class =>
        {
            view.result
        }
        Some(_) => {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "one allocation origin was projected with conflicting element widths",
            ));
        }
        None => new_view_v1(
            origin_index,
            element_width,
            allocation_contract,
            views_by_origin,
            operations,
            next_value,
            ranked_ir,
            scratch,
            resources,
        )?,
    };
    if resources.is_metered() {
        resources.reserve(&mut scratch.comparisons, 2)?;
    } else {
        scratch.comparisons = Vec::with_capacity(2);
    }
    if let Some(precondition) = precondition {
        resources.work(1)?;
        scratch.comparisons.push(precondition);
    }
    resources.work(1)?;
    scratch
        .comparisons
        .push((index, ProductionRankedValueV1::Argument(0)));
    let output_extent = match allocation_provenance.get(receiver).copied().flatten() {
        Some(LocalAllocationProvenanceV1::Argument(argument))
            if checked_success.is_none() && precondition.is_none() =>
        {
            Some(ProductionRankedOutputExtentSourceV1::new(
                argument,
                ProductionRankedValueV1::Local(view),
                ProductionRankedValueV1::Argument(0),
                index,
            ))
        }
        _ => None,
    };
    let comparisons = std::mem::take(&mut scratch.comparisons);
    if resources.is_metered() {
        scratch.access = Some(GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: Vec::new(),
            checked_success,
            comparisons,
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source,
            semantic_site: None,
            output_extent,
        });
        resources.push(
            &mut scratch.access.as_mut().expect("outer access").indices,
            index,
        )?;
    } else {
        scratch.access = Some(GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: vec![index],
            checked_success,
            comparisons,
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source,
            semantic_site: None,
            output_extent,
        });
    }
    if direct_write {
        let slot = direct_write_effects.get_mut(block_index).ok_or(
            ProductionRankedProjectionErrorV1::Unsupported(
                "a write-only access block outside the semantic CFG",
            ),
        )?;
        if slot
            .replace(scratch.access.take().expect("prepared direct access"))
            .is_some()
        {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "multiple write-only effects occupy one semantic block",
            ));
        }
        return Ok(());
    }
    let destination = simple_call_destination(call)?.index() as usize;
    let predicate = option_predicates.get_mut(destination).ok_or(
        ProductionRankedProjectionErrorV1::Unsupported(
            "a checked disjoint destination outside the semantic local table",
        ),
    )?;
    if predicate.is_some() {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "multiple checked predicates for one semantic local",
        ));
    }
    let access = scratch.access.as_ref().expect("prepared checked access");
    if resources.is_metered() {
        scratch.predicate = Some(GuardPredicateV1 {
            comparisons: Vec::new(),
        });
        let cloned = &mut scratch
            .predicate
            .as_mut()
            .expect("outer predicate")
            .comparisons;
        resources.work(access.comparisons.len())?;
        resources.reserve(cloned, access.comparisons.len())?;
        cloned.extend_from_slice(&access.comparisons);
        *predicate = scratch.predicate.take();
        // Growth follows predicate installation, exactly like the original push.
        resources.work(1)?;
        resources.reserve(guarded_accesses, 1)?;
    } else {
        *predicate = Some(GuardPredicateV1::for_access(access));
    }
    guarded_accesses.push(scratch.access.take().expect("prepared checked access"));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_mutable_access_legacy_v1(
    types: &[SemanticTypeDeclV1],
    call: &SemanticDirectCallV1,
    block_index: usize,
    source: SemanticSourceProvenanceV1,
    element: SemanticTypeIdV1,
    index: ProductionRankedValueV1,
    precondition: Option<(ProductionRankedValueV1, ProductionRankedValueV1)>,
    checked_success: Option<ProductionRankedValueV1>,
    direct_write: bool,
    local_allocations: &[Option<AllocationContractV1>],
    allocation_provenance: &[Option<LocalAllocationProvenanceV1>],
    views_by_origin: &mut [Option<ProjectedViewV1>],
    guarded_accesses: &mut Vec<GuardedRankedAccessV1>,
    option_predicates: &mut [Option<GuardPredicateV1>],
    direct_write_effects: &mut [Option<GuardedRankedAccessV1>],
    operations: &mut Vec<ProductionRankedOperationV1>,
    next_value: &mut u32,
    ranked_ir: &mut String,
) -> R<()> {
    append_mutable_access_v1(
        types,
        call,
        block_index,
        source,
        element,
        index,
        precondition,
        checked_success,
        direct_write,
        local_allocations,
        allocation_provenance,
        views_by_origin,
        guarded_accesses,
        option_predicates,
        direct_write_effects,
        operations,
        next_value,
        Some(ranked_ir),
        &mut AccessScratchV1::empty(),
        &mut PreparationResourcesV1::unmetered(),
    )
}

/// DATA only; factory-owned source/context/complete-profile joins are prerequisites.
/// All partial fields remain in rows and the existing S3 operation/predicate owners.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_root_guarded_accesses_v1(
    types: &[SemanticTypeDeclV1],
    callables: &[SemanticCallableDeclV1],
    function: &SemanticFunctionDeclV1,
    index_values: &[Option<ProjectedDisjointIndexV1>],
    option_dominance: &SemanticOptionDominanceV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    local_allocations: &[Option<AllocationContractV1>],
    allocation_provenance: &[Option<LocalAllocationProvenanceV1>],
    option_predicates: &mut [Option<GuardPredicateV1>],
    rows: &mut RootGuardedAccessStorageV1,
    operations: &mut Vec<ProductionRankedOperationV1>,
    next_value: &mut u32,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> R<()> {
    if resources.has_denial() {
        return Err(resource(Resource::Accounting));
    }
    let ledger = resources
        .original_ledger_v1()
        .ok_or_else(|| resource(Resource::Accounting))?;
    resources.work(64)?;
    if rows.ledger.is_some_and(|saved| saved != ledger) {
        return Err(resource(Resource::Accounting));
    }
    if rows.started
        || rows.completed
        || rows.ledger.is_some()
        || rows.frame_credits != 0
        || !rows.views.is_empty()
        || rows.views.capacity() != 0
        || !rows.accesses.is_empty()
        || rows.accesses.capacity() != 0
        || !rows.source_calls.is_empty()
        || rows.source_calls.capacity() != 0
        || !rows.scratch.clear()
    {
        return Err(ProductionRankedProjectionErrorV1::Incomplete(
            "actual guarded access preparation cannot be replaced or retried",
        ));
    }
    let frame = 4096usize
        .checked_add(std::mem::size_of::<RootGuardedAccessStorageV1>())
        .ok_or_else(|| resource(Resource::Arithmetic))?;
    resources.work(frame)?;
    resources.reserve_storage(frame)?;
    rows.frame_credits = frame; // Exact accepted debit, not an inferred type-size report.
    rows.ledger = Some(ledger);
    rows.started = true;
    let locals = function.locals().len();
    if index_values.len() != locals
        || option_predicates.len() != locals
        || local_allocations.len() != locals
        || allocation_provenance.len() != locals
    {
        return Err(ProductionRankedProjectionErrorV1::Incomplete(
            "actual guarded access local tables differ",
        ));
    }
    resources.work(locals)?;
    resources.reserve(&mut rows.views, locals)?;
    rows.views.resize_with(locals, || None);
    let mut source_call_ordinal = 0usize;
    for (block_index, block) in function.blocks().iter().enumerate() {
        resources.work(32)?;
        let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
            continue;
        };
        resources.work(128)?;
        resources.work(16)?;
        let ordinal = source_call_ordinal;
        source_call_ordinal = source_call_ordinal
            .checked_add(1)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
        let Some(SemanticCallableDeclV1::CompilerIntrinsic {
            operation: SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut { element, .. },
            ..
        }) = callables.get(call.callee().index() as usize)
        else {
            continue;
        };
        if let Some(SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) =
            call.arguments().get(1)
        {
            resources.work(place.projections().len())?;
        }
        let projected = identity_operand_v1(
            call,
            index_values,
            option_dominance,
            enum_payload_dominance,
            block_index,
        )?;
        // Reserve OUTER association storage before the fallible append. The
        // successful append is followed by a nonallocating Copy push; failed
        // partial emissions retain both owners and never receive a completed loan.
        resources.work(32)?;
        resources.reserve(&mut rows.source_calls, 1)?;
        let guarded_access = rows.accesses.len();
        append_mutable_access_v1(
            types,
            call,
            block_index,
            block.terminator().source(),
            *element,
            projected.value,
            projected.precondition,
            None,
            false,
            local_allocations,
            allocation_provenance,
            &mut rows.views,
            &mut rows.accesses,
            option_predicates,
            &mut [],
            operations,
            next_value,
            None,
            &mut rows.scratch,
            resources,
        )?;
        // Destination lookup stays after the append to preserve its existing
        // semantic refusal order. Success already required a simple destination.
        rows.source_calls.push(RootGuardedSourceCallV1 {
            source_call_ordinal: ordinal,
            block: block_index,
            callee: call.callee(),
            destination: call
                .destination()
                .expect("successful accessor destination")
                .place()
                .local(),
            guarded_access,
        });
    }
    if resources.has_denial() || resources.original_ledger_v1() != Some(ledger) {
        return Err(resource(Resource::Accounting));
    }
    rows.completed = true; // Access payload DATA only, never origins/recipe/admission.
    Ok(())
}
