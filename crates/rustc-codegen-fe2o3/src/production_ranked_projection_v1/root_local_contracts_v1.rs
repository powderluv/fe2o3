//! Borrowed local-contract DATA. Raw part constructors do not establish owner,
//! source-custody, completed block-stream, capability or admission authority.
use super::bf16_nominal_preparation_resources_v1::{PreparationResourcesV1, resource};
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceErrorV1 as Resource, CanonicalKernelIrWorkLedgerIdentityV1,
};

pub(super) type LocalContractLedgerV1 = (usize, CanonicalKernelIrWorkLedgerIdentityV1);

/// Exact old short-circuit predicate; Legacy still constructs its old Vec<bool>.
/// Actual queries reuse this predicate over borrowed rich rows without a clone.
pub(super) fn immutable_candidate_v1(
    function: &SemanticFunctionDeclV1,
    local: usize,
    counts: &[u8],
    assignments: &[Option<ScalarAssignmentSiteV1>],
    address_escaped: &[bool],
) -> bool {
    !function.locals()[local].role().is_entry_argument()
        && counts[local] == 1
        && assignments[local].is_some()
        && !address_escaped[local]
}

enum ImmutableInputsV1<'a> {
    Legacy(&'a [bool]),
    Source {
        function: &'a SemanticFunctionDeclV1,
        counts: &'a [u8],
        assignments: &'a [Option<ScalarAssignmentSiteV1>],
        address_escaped: &'a [bool],
    },
}

/// Raw inert parts used only by the actual factory and synthetic hostile controls.
/// Same-function/length checks are not promoted into an authenticated owner.
pub(super) struct SourceLocalContractPartsV1<'a> {
    pub(super) function: &'a SemanticFunctionDeclV1,
    pub(super) table_function: &'a SemanticFunctionDeclV1,
    pub(super) counts: &'a [u8],
    pub(super) assignments: &'a [Option<ScalarAssignmentSiteV1>],
    pub(super) address_escaped: &'a [bool],
    pub(super) allocations: &'a [Option<AllocationContractV1>],
    pub(super) allocation_provenance: &'a [Option<LocalAllocationProvenanceV1>],
    pub(super) origins: &'a [Option<CheckedReferenceOriginV1>],
    pub(super) option_dominance: &'a SemanticOptionDominanceV1,
    pub(super) enum_payload_dominance: &'a SemanticEnumPayloadDominanceV1,
}

/// Borrowed query decisions shared with ordinary projection; no mutable tables.
pub(super) struct BorrowedLocalContractsV1<'a> {
    immutable: ImmutableInputsV1<'a>,
    allocations: &'a [Option<AllocationContractV1>],
    allocation_provenance: &'a [Option<LocalAllocationProvenanceV1>],
    origins: &'a [Option<CheckedReferenceOriginV1>],
    option_dominance: &'a SemanticOptionDominanceV1,
    enum_payload_dominance: &'a SemanticEnumPayloadDominanceV1,
}

impl<'a> BorrowedLocalContractsV1<'a> {
    pub(super) fn legacy(owned: &'a ProjectionLocalContractsV1) -> Self {
        Self {
            immutable: ImmutableInputsV1::Legacy(&owned.immutable_locals),
            allocations: &owned.allocations,
            allocation_provenance: &owned.allocation_provenance,
            origins: &owned.checked_references.origins,
            option_dominance: &owned.checked_references.option_dominance,
            enum_payload_dominance: &owned.checked_references.enum_payload_dominance,
        }
    }

    pub(super) fn source_data(
        parts: SourceLocalContractPartsV1<'a>,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> Result<Self, ProductionRankedProjectionErrorV1> {
        if resources.has_denial() || resources.original_ledger_v1().is_none() {
            return Err(resource(Resource::Accounting));
        }
        // Only fixed identity/length checks; no hidden scan or table allocation.
        resources.work(128)?;
        root_checked_references_v1::require_same_source_v1(parts.function, parts.table_function)?;
        let locals = parts.function.locals().len();
        if [
            parts.counts.len(),
            parts.assignments.len(),
            parts.address_escaped.len(),
            parts.allocations.len(),
            parts.allocation_provenance.len(),
            parts.origins.len(),
        ]
        .iter()
        .any(|length| *length != locals)
        {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "borrowed local-contract table cardinality differs",
            ));
        }
        Ok(Self {
            immutable: ImmutableInputsV1::Source {
                function: parts.function,
                counts: parts.counts,
                assignments: parts.assignments,
                address_escaped: parts.address_escaped,
            },
            allocations: parts.allocations,
            allocation_provenance: parts.allocation_provenance,
            origins: parts.origins,
            option_dominance: parts.option_dominance,
            enum_payload_dominance: parts.enum_payload_dominance,
        })
    }

    pub(super) fn immutable(&self, local: usize) -> bool {
        match &self.immutable {
            ImmutableInputsV1::Legacy(values) => values.get(local) == Some(&true),
            ImmutableInputsV1::Source {
                function,
                counts,
                assignments,
                address_escaped,
            } => {
                function.locals().get(local).is_some()
                    && immutable_candidate_v1(function, local, counts, assignments, address_escaped)
            }
        }
    }

    pub(super) fn allocation(&self, local: usize) -> Option<AllocationContractV1> {
        self.allocations.get(local).copied().flatten()
    }

    pub(super) fn allocation_provenance(
        &self,
        local: usize,
    ) -> Option<LocalAllocationProvenanceV1> {
        self.allocation_provenance.get(local).copied().flatten()
    }

    pub(super) fn origin(
        &self,
        place: &SemanticPlaceV1,
        block: usize,
    ) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
        root_checked_references_v1::origin_for_slices_v1(
            place,
            block,
            self.origins,
            self.option_dominance,
            self.enum_payload_dominance,
        )
    }

    /// DATA query on the original lexical account. Only the outer actual factory
    /// can attach this view to an authenticated S5A loan.
    pub(super) fn query_on_ledger(
        &self,
        function: &SemanticFunctionDeclV1,
        place: &SemanticPlaceV1,
        block: usize,
        expected: LocalContractLedgerV1,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> Result<LocalContractDecisionV1, ProductionRankedProjectionErrorV1> {
        require_local_contract_ledger_v1(expected, resources)?;
        let work = place
            .projections()
            .len()
            .checked_add(80)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
        resources.work(work)?;
        let ImmutableInputsV1::Source {
            function: actual, ..
        } = &self.immutable
        else {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "actual local-contract query requires borrowed source rows",
            ));
        };
        root_checked_references_v1::require_same_source_v1(function, actual)?;
        let local = place.local().index() as usize;
        if function.locals().get(local).is_none()
            || function.blocks().get(block).is_none()
            || u32::try_from(block).is_err()
        {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "actual local-contract query source coordinate differs",
            ));
        }
        // Preserve checked-reference refusal before inspecting fallback decisions.
        let origin = self.origin(place, block)?;
        Ok(LocalContractDecisionV1 {
            origin,
            immutable: self.immutable(local),
            allocation: self.allocation(local),
            allocation_provenance: self.allocation_provenance(local),
        })
    }
}

/// These are existing local fallback decisions, NOT a projected access/effect.
pub(super) struct LocalContractDecisionV1 {
    pub(super) origin: Option<CheckedReferenceSourceV1>,
    pub(super) immutable: bool,
    pub(super) allocation: Option<AllocationContractV1>,
    pub(super) allocation_provenance: Option<LocalAllocationProvenanceV1>,
}

pub(super) fn require_local_contract_ledger_v1(
    expected: LocalContractLedgerV1,
    resources: &PreparationResourcesV1<'_, '_>,
) -> Result<(), ProductionRankedProjectionErrorV1> {
    if resources.has_denial() || resources.original_ledger_v1() != Some(expected) {
        return Err(resource(Resource::Accounting));
    }
    Ok(())
}

/// Prepaid wrapper/callback/query-value storage only. It does not allocate a
/// table and is retained by the existing outer resource owner until teardown.
pub(super) fn local_contract_frame_v1<R, F>(
    actual_view_bytes: usize,
) -> Result<usize, ProductionRankedProjectionErrorV1> {
    let mut frame = 4096usize;
    for amount in [
        actual_view_bytes.checked_mul(2),
        Some(std::mem::size_of::<SourceLocalContractPartsV1<'static>>()),
        std::mem::size_of::<BorrowedLocalContractsV1<'static>>().checked_mul(2),
        std::mem::size_of::<LocalContractDecisionV1>().checked_mul(2),
        std::mem::size_of::<LocalContractLedgerV1>().checked_mul(2),
        std::mem::size_of::<F>().checked_mul(2),
        std::mem::size_of::<Result<R, ProductionRankedProjectionErrorV1>>().checked_mul(2),
    ] {
        frame = frame
            .checked_add(amount.ok_or_else(|| resource(Resource::Arithmetic))?)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
    }
    Ok(frame)
}
