//! Borrowed semantic dereference decision for the existing owned ordinary table.
//! No new origin producer, authenticated-edge graph, or nominal admission.
//! The strict resource context uses only the exact-function identity check.
use super::*;

struct BorrowedCheckedReferencesV1<'a> {
    origins: &'a [Option<CheckedReferenceOriginV1>],
    option_dominance: &'a SemanticOptionDominanceV1,
    enum_payload_dominance: &'a SemanticEnumPayloadDominanceV1,
}

pub(super) fn require_same_source_v1(
    actual: &SemanticFunctionDeclV1,
    candidate: &SemanticFunctionDeclV1,
) -> Result<(), ProductionRankedProjectionErrorV1> {
    if !std::ptr::eq(actual, candidate) {
        return Err(ProductionRankedProjectionErrorV1::Incomplete(
            "nominal recipe tables do not borrow the actual source function",
        ));
    }
    Ok(())
}

pub(super) fn origin_for_owned_v1(
    place: &SemanticPlaceV1,
    block_index: usize,
    references: &CheckedReferencesV1,
) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
    let borrowed = BorrowedCheckedReferencesV1 {
        origins: &references.origins,
        option_dominance: &references.option_dominance,
        enum_payload_dominance: &references.enum_payload_dominance,
    };
    origin_for_borrowed_v1(place, block_index, &borrowed)
}

/// Inert borrowed-slice adapter; the single original availability body stays authoritative.
pub(super) fn origin_for_slices_v1(
    place: &SemanticPlaceV1,
    block_index: usize,
    origins: &[Option<CheckedReferenceOriginV1>],
    option_dominance: &SemanticOptionDominanceV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
    let borrowed = BorrowedCheckedReferencesV1 {
        origins,
        option_dominance,
        enum_payload_dominance,
    };
    origin_for_borrowed_v1(place, block_index, &borrowed)
}

fn origin_for_borrowed_v1(
    place: &SemanticPlaceV1,
    block_index: usize,
    references: &BorrowedCheckedReferencesV1<'_>,
) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
    let Some(origin) = checked_reference_origin_for_place(place, references.origins) else {
        return Ok(None);
    };
    if !origin.availability.is_none_or(|availability| {
        capability_availability_allows(
            references.option_dominance,
            references.enum_payload_dominance,
            availability,
            SemanticBlockIdV1::from_index(block_index as u32),
        )
    }) {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "a checked reference is dereferenced outside its authenticated payload region",
        ));
    }
    Ok(Some(origin.source))
}

/// Paid read-only decision over tables borrowed by the actual S5A factory.
/// Raw callers can obtain only inert data, never a source/ready constructor.
#[allow(clippy::too_many_arguments)]
pub(super) fn origin_for_actual_borrowed_v1(
    function: &SemanticFunctionDeclV1,
    table_function: &SemanticFunctionDeclV1,
    place: &SemanticPlaceV1,
    block_index: usize,
    origins: &[Option<CheckedReferenceOriginV1>],
    option_dominance: &SemanticOptionDominanceV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    resources: &mut super::bf16_nominal_preparation_resources_v1::PreparationResourcesV1<'_, '_>,
) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
    use super::bf16_nominal_preparation_resources_v1::resource;
    use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1 as Resource;
    if resources.has_denial() || resources.original_ledger_v1().is_none() {
        return Err(resource(Resource::Accounting));
    }
    let work = place
        .projections()
        .len()
        .checked_add(64)
        .ok_or_else(|| resource(Resource::Arithmetic))?;
    resources.work(work)?;
    require_same_source_v1(function, table_function)?;
    if origins.len() != function.locals().len()
        || function.blocks().get(block_index).is_none()
        || u32::try_from(block_index).is_err()
    {
        return Err(ProductionRankedProjectionErrorV1::Incomplete(
            "actual borrowed use source table or block differs",
        ));
    }
    let borrowed = BorrowedCheckedReferencesV1 {
        origins,
        option_dominance,
        enum_payload_dominance,
    };
    // The shared original availability decision is O(1) after the paid place
    // projection walk (retained dominance intervals, no new graph allocation).
    origin_for_borrowed_v1(place, block_index, &borrowed)
}
