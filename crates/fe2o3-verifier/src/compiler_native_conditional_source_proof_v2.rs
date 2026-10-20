//! Independent conditional source/N custody under explicit external policies.
//!
//! This consumer replays content, not compiler/registration/launch origin. It
//! retains the actual reconstructed source and pending conditional arenas, not
//! ordinary clean evidence, final F, machine admission or a host capability.

use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use fe2o3_lower_mir_kernel::{ProductionConditionalRootInputV1, ReplayedNativeSourceV1};
use std::mem::size_of;

use crate::compiler_native_conditional_source_packet_v2::{
    NativeConditionalSourcePacketInputV2, NativeConditionalSourceRootV2,
    with_decoded_native_conditional_source_packet_v2,
};
use crate::{ProductionConditionalFormulaReportV2, RetainedProductionConditionalFormulaV2};

#[path = "compiler_native_conditional_source_proof_v2/account.rs"]
mod account;
#[path = "compiler_native_conditional_source_proof_v2/cpu_origin.rs"]
mod cpu_origin;
pub use cpu_origin::{NativeConditionalCpuExpectationV1, NativeConditionalCpuOriginExpectationV1};
#[path = "compiler_native_conditional_source_proof_v2/cpu_mapping.rs"]
pub(crate) mod cpu_mapping;
pub use cpu_mapping::{
    NativeConditionalCpuMappingContextV1, NativeConditionalCpuMappingExpectationV1,
};
#[path = "compiler_native_conditional_source_proof_v2/error.rs"]
mod error;
#[path = "compiler_native_conditional_final_v2.rs"]
pub(crate) mod final_replay;
#[path = "compiler_native_conditional_source_proof_v2/reconstruct.rs"]
mod reconstruct;
#[path = "compiler_native_conditional_source_proof_v2/root.rs"]
mod root;
#[path = "compiler_native_conditional_source_proof_v2/selection.rs"]
mod selection;

use error::Cause;
pub use error::NativeConditionalSourceProofErrorV2;
use error::NativeConditionalSourceProofErrorV2 as E;
pub use selection::select_native_conditional_ownership_site_v2;

/// Independently accepted policies in complete semantic-root order.
/// Transported receipt keys never select the accepted signer roster.
pub struct NativeConditionalRootPolicyV2<'a> {
    /// Actual semantic root ordinal, not canonical kernel order.
    pub semantic_root: u32,
    /// Accepted effect signers and exact toolchain for existing staging.
    pub effects: &'a fe2o3_pliron::ProductionRefinementStagingPolicyV2,
    /// Accepted conditional V2 signer, toolchain and boundary.
    pub formula: &'a fe2o3_functional_proof::FunctionalRefinementImportPolicyV2,
}

struct ReplayedRoot {
    input: ProductionConditionalRootInputV1,
    formula: RetainedProductionConditionalFormulaV2,
}

/// Move-only content reconstruction; all arenas and proof owners remain private.
/// Origin of registration, logical names and launch max_grid remains external.
/// There is no ordinary receipt conversion or final-F/native publication path.
///
/// ```compile_fail
/// use fe2o3_verifier::ReplayedNativeConditionalSourceV2;
/// fn duplicate(value: ReplayedNativeConditionalSourceV2) { let _ = value.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::{ReplayedNativeConditionalSourceV2, ValidatedNativeCompilerRankedSourceProofV1};
/// fn ordinary(value: ReplayedNativeConditionalSourceV2) -> ValidatedNativeCompilerRankedSourceProofV1 {
///     value.into()
/// }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::{ReplayedNativeConditionalSourceV2, ProductionConditionalFormulaReportV2};
/// fn install(report: ProductionConditionalFormulaReportV2) -> ReplayedNativeConditionalSourceV2 {
///     report.into()
/// }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::ReplayedNativeConditionalSourceV2;
/// use fe2o3_lower_mir_kernel::ProductionConditionalRootInputV1;
/// fn arena(value: &ReplayedNativeConditionalSourceV2) -> &ProductionConditionalRootInputV1 {
///     &value.roots[0].input
/// }
/// ```
#[must_use = "reserve the accompanying storage before further controlled allocation"]
pub struct ReplayedNativeConditionalSourceV2 {
    source: ReplayedNativeSourceV1,
    roots: Vec<ReplayedRoot>,
    canonical_kernel_order: Vec<u32>,
}
impl ReplayedNativeConditionalSourceV2 {
    /// Borrows normal source/N reconstruction, not an original compiler witness.
    pub fn source(&self) -> &ReplayedNativeSourceV1 {
        &self.source
    }
    /// Complete conditional root roster in source order.
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }
    /// Source ordinals sorted by the actual complete descriptor KernelId.
    pub fn canonical_kernel_order(&self) -> &[u32] {
        &self.canonical_kernel_order
    }
    /// Inert identities from this root's retained, strictly imported V2 proof.
    pub fn formula_report(&self, ordinal: usize) -> Option<ProductionConditionalFormulaReportV2> {
        self.roots.get(ordinal).map(|root| root.formula.report())
    }
}

/// Exact declared retained source/arena/buffer and wrapper capacity reservation.
/// This is UNRESERVED on return; it is not proof or authenticated ledger custody.
/// Inherited semantic/Pliron engines retain their existing bounded domains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConditionalSourceStorageV2(usize);
impl NativeConditionalSourceStorageV2 {
    /// Reserve before further controlled allocation; drop custody before refund.
    pub const fn retained_storage(self) -> usize {
        self.0
    }
}

/// Reconstructs the complete conditional source roster using the original caller
/// account. Every root uses one decoded CPU scope and one strict formula import
/// inside the actual lower request callback. No protected proof is executed.
/// Packet, CPU, lower and formula postchecks finish before returning custody.
/// The returned storage is unreserved, following existing native replay APIs.
/// This does not authenticate source/registration/launch origin or enable F.
/// Opaque inherited CPU failures and unwinds retain terminal reservations: they
/// cannot establish that every nested floor survived, even after owners drop.
pub fn validate_native_conditional_source_packet_v2(
    bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    account::transfer(budget, |budget| {
        with_decoded_native_conditional_source_packet_v2(bytes, budget, |packet, budget| {
            reconstruct::reconstruct(packet, accepted, budget)
        })
        .map_err(|error| E(Cause::Packet(error)))?
    })
}

/// Explicit origin-selected counterpart; the original entrypoint remains V1-only.
/// The complete ordered expectations are independently supplied, caller-owned
/// and prepaid. Agreement does not authenticate their original Loan provenance.
pub fn validate_native_conditional_source_packet_with_cpu_origins_v2(
    bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    let origins = cpu_origin::CpuReplayMode::Expected(expected_cpu);
    account::transfer(budget, |budget| {
        origins.require_backing(budget)?;
        with_decoded_native_conditional_source_packet_v2(bytes, budget, |packet, budget| {
            reconstruct::reconstruct_with_origins(packet, accepted, origins, budget)
        })
        .map_err(|error| E(Cause::Packet(error)))?
    })
}

#[cfg(test)]
#[path = "compiler_native_conditional_source_proof_v2/tests.rs"]
mod tests;

/// Replays a complete inventory projection against the actual reconstructed source.
/// This lower content API does not authenticate externally supplied inventory or
/// expectations. Caller prepays their actual backing and view headers. The
/// original registration-only and origin-only entrypoints remain unchanged.
pub fn validate_native_conditional_source_packet_with_cpu_mapping_v2(
    bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    mapping: &NativeConditionalCpuMappingContextV1<'_>,
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    let selection = cpu_mapping::ReplaySelection::Mapping(mapping);
    account::transfer(budget, |budget| {
        selection.require_backing(budget)?;
        with_decoded_native_conditional_source_packet_v2(bytes, budget, |packet, budget| {
            reconstruct::reconstruct_selected(packet, accepted, selection, budget)
        })
        .map_err(|error| E(Cause::Packet(error)))?
    })
}
