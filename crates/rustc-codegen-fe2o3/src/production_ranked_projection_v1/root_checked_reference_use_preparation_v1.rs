//! Private source-use DATA and shared ordinary checked-reference site mechanics.
//! A borrowed source occurrence has no operation cursor, completed block-stream,
//! checked-ready token, recipe or admission authority. Paid site storage below is
//! a component only: a future same-owner block projector must own its vectors.
use super::bf16_nominal_preparation_resources_v1::{PreparationResourcesV1, resource};
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1 as Resource;

type UseResult<T> = Result<T, ProductionRankedProjectionErrorV1>;

pub(super) struct SourceUseOccurrenceV1<'a> {
    pub(super) place: &'a SemanticPlaceV1,
    pub(super) site: ProjectedSemanticAccessSiteV1,
    pub(super) ordinal: usize,
    pub(super) access: AccessKindAttr,
    pub(super) atomic: Option<SemanticAtomicAccessV1>,
    pub(super) requirement: PlaceAccessRequirementV1,
    pub(super) source: SemanticSourceProvenanceV1,
}
struct SelectorV1<'a> {
    wanted: usize,
    visited: usize,
    selected: Option<SourceUseOccurrenceV1<'a>>,
    site: ProjectedSemanticAccessSiteV1,
    source: SemanticSourceProvenanceV1,
}
impl<'a> SelectorV1<'a> {
    fn place(
        &mut self,
        place: &'a SemanticPlaceV1,
        access: AccessKindAttr,
        atomic: Option<SemanticAtomicAccessV1>,
        requirement: PlaceAccessRequirementV1,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> UseResult<()> {
        resources.work(32)?;
        let ordinal = self.visited;
        self.visited = self
            .visited
            .checked_add(1)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
        if ordinal == self.wanted {
            self.selected = Some(SourceUseOccurrenceV1 {
                place,
                site: self.site,
                ordinal,
                access,
                atomic,
                requirement,
                source: self.source,
            });
        }
        Ok(())
    }
    fn operand(
        &mut self,
        operand: &'a SemanticOperandV1,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> UseResult<()> {
        resources.work(8)?;
        match operand {
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => self.place(
                place,
                AccessKindAttr::Read,
                None,
                PlaceAccessRequirementV1::IfMemory,
                resources,
            ),
            SemanticOperandV1::Constant(_) => Ok(()),
        }
    }
    fn rvalue(
        &mut self,
        value: &'a SemanticRvalueKindV1,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> UseResult<()> {
        resources.work(16)?;
        match value {
            SemanticRvalueKindV1::Use(operand)
            | SemanticRvalueKindV1::Unary { operand, .. }
            | SemanticRvalueKindV1::Cast { operand, .. } => self.operand(operand, resources),
            SemanticRvalueKindV1::Binary { left, right, .. } => {
                self.operand(left, resources)?;
                self.operand(right, resources)
            }
            SemanticRvalueKindV1::CheckedBinary(binary) => {
                self.operand(binary.left(), resources)?;
                self.operand(binary.right(), resources)
            }
            SemanticRvalueKindV1::UncheckedBinary(binary) => {
                self.operand(binary.left(), resources)?;
                self.operand(binary.right(), resources)
            }
            SemanticRvalueKindV1::Aggregate(aggregate) => {
                for operand in aggregate.operands() {
                    self.operand(operand, resources)?;
                }
                Ok(())
            }
            SemanticRvalueKindV1::Load(load) => self.place(
                load.source(),
                if load.atomic().is_some() {
                    AccessKindAttr::AtomicRead
                } else {
                    AccessKindAttr::Read
                },
                load.atomic(),
                PlaceAccessRequirementV1::ExplicitMemory,
                resources,
            ),
            // The ordinary projector separately validates address formation.
            // It is not a value access, and this diagnostic selector cannot
            // certify those other allocation/bounds requirements.
            SemanticRvalueKindV1::AddressOf { .. } | SemanticRvalueKindV1::Borrow { .. } => Ok(()),
            SemanticRvalueKindV1::Length(place) | SemanticRvalueKindV1::Discriminant(place) => self
                .place(
                    place,
                    AccessKindAttr::Read,
                    None,
                    PlaceAccessRequirementV1::IfMemory,
                    resources,
                ),
        }
    }
}

/// Select one actual source place occurrence, not a projected memory operation.
/// Every candidate comes from the borrowed function; caller-supplied places,
// access kinds, provenance and operation cursors are not accepted. Calls stay
/// deliberately incomplete until the separate exact call/effect router is ready.
pub(super) fn select_source_use_v1<'a>(
    function: &'a SemanticFunctionDeclV1,
    site: ProjectedSemanticAccessSiteV1,
    ordinal: usize,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> UseResult<SourceUseOccurrenceV1<'a>> {
    if resources.has_denial() {
        return Err(resource(Resource::Accounting));
    }
    resources.work(64)?;
    let block =
        function
            .blocks()
            .get(site.block)
            .ok_or(ProductionRankedProjectionErrorV1::Unsupported(
                "actual source use block absent",
            ))?;
    let statement = match site.statement {
        Some(index) => Some(block.statements().get(index).ok_or(
            ProductionRankedProjectionErrorV1::Unsupported("actual source use statement absent"),
        )?),
        None => None,
    };
    let mut selector = SelectorV1 {
        wanted: ordinal,
        visited: 0,
        selected: None,
        site,
        source: statement.map_or_else(|| block.terminator().source(), |s| s.source()),
    };
    if let Some(statement) = statement {
        match statement.kind() {
            SemanticStatementKindV1::Assign(assignment) => {
                selector.rvalue(assignment.value().kind(), resources)?;
                selector.place(
                    assignment.destination(),
                    AccessKindAttr::Write,
                    None,
                    PlaceAccessRequirementV1::IfMemory,
                    resources,
                )?;
            }
            SemanticStatementKindV1::Store(store) => {
                selector.operand(store.value(), resources)?;
                selector.place(
                    store.destination(),
                    if store.atomic().is_some() {
                        AccessKindAttr::AtomicWrite
                    } else {
                        AccessKindAttr::Write
                    },
                    store.atomic(),
                    PlaceAccessRequirementV1::ExplicitMemory,
                    resources,
                )?;
            }
            SemanticStatementKindV1::AtomicRmw(atomic) => {
                selector.operand(atomic.value(), resources)?;
                selector.place(
                    atomic.address(),
                    AccessKindAttr::AtomicReadModifyWrite,
                    Some(atomic.access()),
                    PlaceAccessRequirementV1::ExplicitMemory,
                    resources,
                )?;
                selector.place(
                    atomic.destination(),
                    AccessKindAttr::Write,
                    None,
                    PlaceAccessRequirementV1::IfMemory,
                    resources,
                )?;
            }
            SemanticStatementKindV1::AtomicCompareExchange(atomic) => {
                // Preserve the real ordinary order, not an assumed RHS-first rule.
                selector.place(
                    atomic.destination(),
                    AccessKindAttr::Write,
                    None,
                    PlaceAccessRequirementV1::IfMemory,
                    resources,
                )?;
                selector.place(
                    atomic.address(),
                    AccessKindAttr::AtomicReadModifyWrite,
                    Some(atomic.success()),
                    PlaceAccessRequirementV1::ExplicitMemory,
                    resources,
                )?;
                selector.operand(atomic.expected(), resources)?;
                selector.operand(atomic.replacement(), resources)?;
            }
            SemanticStatementKindV1::SetDiscriminant { place, .. }
            | SemanticStatementKindV1::Deinitialize(place) => selector.place(
                place,
                AccessKindAttr::Write,
                None,
                PlaceAccessRequirementV1::IfMemory,
                resources,
            )?,
            SemanticStatementKindV1::StorageLive(local)
            | SemanticStatementKindV1::StorageDead(local) => {
                if function.locals().get(local.index() as usize).is_none() {
                    return Err(ProductionRankedProjectionErrorV1::Unsupported(
                        "a storage statement with an out-of-range local",
                    ));
                }
            }
            SemanticStatementKindV1::Assume(condition) => selector.operand(condition, resources)?,
            SemanticStatementKindV1::Nop => {}
        }
    } else {
        match block.terminator().kind() {
            SemanticTerminatorKindV1::SwitchInt { discriminant, .. } => {
                selector.operand(discriminant, resources)?
            }
            SemanticTerminatorKindV1::Assert { condition, .. } => {
                selector.operand(condition, resources)?
            }
            SemanticTerminatorKindV1::Call(_)
            | SemanticTerminatorKindV1::TailCall(_)
            | SemanticTerminatorKindV1::Drop { .. } => {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "actual source use at a call or drop requires full access/effect routing",
                ));
            }
            SemanticTerminatorKindV1::Goto(_)
            | SemanticTerminatorKindV1::FalseEdge { .. }
            | SemanticTerminatorKindV1::Return
            | SemanticTerminatorKindV1::UnwindResume
            | SemanticTerminatorKindV1::UnwindTerminate
            | SemanticTerminatorKindV1::Abort
            | SemanticTerminatorKindV1::Unreachable => {}
        }
    }
    selector
        .selected
        .ok_or(ProductionRankedProjectionErrorV1::Unsupported(
            "actual source use occurrence absent",
        ))
}

/// Shared checked-reference branch. The ordinary adapter uses its unchanged
/// unmetered clone/reserve order. Paid callers own all partial site slots through
/// their enclosing postflights; this function never refunds or grants a loan.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_checked_reference_site_v1(
    origin: CheckedReferenceSourceV1,
    access: AccessKindAttr,
    atomic: Option<SemanticAtomicAccessV1>,
    source: SemanticSourceProvenanceV1,
    guarded_accesses: &[GuardedRankedAccessV1],
    guarded_sites: &mut Vec<GuardedAccessSiteV1>,
    operations: &[ProductionRankedOperationV1],
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> UseResult<()> {
    if resources.has_denial() {
        return Err(resource(Resource::Accounting));
    }
    resources.work(32)?;
    if resources.is_metered() && access.is_atomic() != atomic.is_some() {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "an atomic access whose ordering/scope contract is missing or attached to a non-atomic access",
        ));
    }
    match origin {
        CheckedReferenceSourceV1::GuardedAccess(origin) => {
            if atomic.is_some() {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "an atomic access through a checked disjoint reference before exact atomic capability projection",
                ));
            }
            if resources.is_metered() {
                let guarded = guarded_accesses.get(origin).ok_or(
                    ProductionRankedProjectionErrorV1::Unsupported(
                        "a checked disjoint reference whose access origin is out of range",
                    ),
                )?;
                resources.work(1)?;
                resources.reserve(guarded_sites, 1)?;
                let index = guarded_sites.len();
                // Install the physical outer slot BEFORE either nested allocation.
                guarded_sites.push(GuardedAccessSiteV1 {
                    insertion_operation: operations.len(),
                    access: GuardedRankedAccessV1 {
                        view: guarded.view,
                        indices: Vec::new(),
                        comparisons: Vec::new(),
                        checked_success: guarded.checked_success,
                        access,
                        memory_space: guarded.memory_space,
                        source,
                        semantic_site: guarded.semantic_site,
                        output_extent: guarded.output_extent,
                    },
                });
                let retained = &mut guarded_sites[index].access;
                resources.work(guarded.indices.len())?;
                resources.reserve(&mut retained.indices, guarded.indices.len())?;
                retained.indices.extend_from_slice(&guarded.indices);
                resources.work(guarded.comparisons.len())?;
                resources.reserve(&mut retained.comparisons, guarded.comparisons.len())?;
                retained.comparisons.extend_from_slice(&guarded.comparisons);
                return Ok(());
            }
            let mut guarded = guarded_accesses.get(origin).cloned().ok_or(
                ProductionRankedProjectionErrorV1::Unsupported(
                    "a checked disjoint reference whose access origin is out of range",
                ),
            )?;
            guarded.access = access;
            guarded.source = source;
            guarded_sites.try_reserve(1).map_err(|_| {
                ProductionRankedProjectionErrorV1::Unsupported(
                    "checked disjoint access-site storage cannot be reserved",
                )
            })?;
            guarded_sites.push(GuardedAccessSiteV1 {
                insertion_operation: operations.len(),
                access: guarded,
            });
            Ok(())
        }
        CheckedReferenceSourceV1::ProjectedSharedBorrow
            if atomic.is_none() && access == AccessKindAttr::Read =>
        {
            Ok(())
        }
        CheckedReferenceSourceV1::ProjectedSharedBorrow => {
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a projected shared reference used for a non-read memory effect",
            ))
        }
    }
}

/// A shared slice mutation, not an actual-source or completed-stream constructor.
/// The ordinary caller supplies only slices appended during its actual site.
/// A paid future owner must keep partially bound slices on any error or unwind.
pub(super) fn bind_projected_access_site_with_resources_v1(
    sources: &mut [ProjectedAccessSourceV1],
    guarded_sites: &mut [GuardedAccessSiteV1],
    site: ProjectedSemanticAccessSiteV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> UseResult<()> {
    if resources.has_denial() {
        return Err(resource(Resource::Accounting));
    }
    for source in sources {
        resources.work(16)?;
        if source.semantic_site.replace(site).is_some() {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a projected access was attributed to multiple semantic sites",
            ));
        }
    }
    for guarded in guarded_sites {
        resources.work(16)?;
        if guarded.access.semantic_site.replace(site).is_some() {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a guarded access was attributed to multiple semantic sites",
            ));
        }
    }
    Ok(())
}
