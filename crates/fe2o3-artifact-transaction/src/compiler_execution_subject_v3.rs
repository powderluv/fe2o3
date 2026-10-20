//! Conditional V5 content binding, not protected compiler execution evidence.
use super::{CompilerExecutionSubjectErrorV1, InertCompilerExecutionContentBindingV1, codec};
use crate::{
    BuildAttempt, CompilerModuleHandoffReceiptV5, CompilerModuleHandoffSlotV5,
    CompilerModuleHandoffTransactionIdentityV5, ConsumedCompilerModuleHandoffV5,
    MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5,
};
use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_compiler_ffi::{
    INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5 as METADATA,
    InertSemanticCompilerModuleHandoffV5 as Handoff,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use std::{fmt, mem::size_of};

type Binding = InertCompilerExecutionContentBindingV1;

pub const INERT_COMPILER_EXECUTION_SUBJECT_MAGIC_V3: [u8; 8] = *b"F2O3CES3";
pub const INERT_COMPILER_EXECUTION_SUBJECT_VERSION_V3: u16 = 3;
pub const INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V3: usize = codec::BYTES;
const SCHEMA: codec::Schema = codec::Schema {
    magic: INERT_COMPILER_EXECUTION_SUBJECT_MAGIC_V3,
    version: INERT_COMPILER_EXECUTION_SUBJECT_VERSION_V3,
    identity_domain: b"FE2O3/INERT-COMPILER-EXECUTION-SUBJECT/V3\0",
    transaction_label: "V5 handoff transaction",
    outer_label: "outer V5 handoff",
};
const RETAINED: usize = size_of::<InertCompilerExecutionSubjectV3>()
    + size_of::<InertCompilerExecutionSubjectStorageV3>();
/// Fixed logical work for cached-field extraction, closure validation, two
/// subject hashes, reconstruction and comparison, including entry work.
/// No invocation or payload serialization occurs; this is not instruction count.
pub const INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3: usize = 32 * codec::BYTES + 4096 + 8;
/// Fixed additional logical peak, including result/receipt, staging, hashes and
/// scalar/error scratch. Not an allocator/RSS bound or a payload reservation.
pub const INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3: usize = 4 * RETAINED
    + 4 * size_of::<codec::Fields>()
    + 4 * codec::BYTES
    + 2 * size_of::<sha2::Sha256>()
    + 4096;

const fn result_overhead<P>() -> usize {
    4 * (size_of::<Result<P>>() - size_of::<P>())
        + crate::compiler_module_handoff::resources::fixed_scope_overhead::<Result<P>>()
}
const SCALAR_STORAGE: usize = codec::SCALAR_STORAGE
    + 2 * size_of::<CompilerModuleHandoffReceiptV5>()
    + 64 * size_of::<usize>();
const _: () = {
    type Output = (
        InertCompilerExecutionSubjectV3,
        InertCompilerExecutionSubjectStorageV3,
    );
    assert!(codec::BYTES == 690);
    assert!(size_of::<Output>() <= RETAINED);
    assert!(SCALAR_STORAGE + result_overhead::<Output>() <= 4096);
    assert!(SCALAR_STORAGE + result_overhead::<bool>() <= 4096);
};

/// Distinct V3 subject identity, not a V1/V2 execution attestation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InertCompilerExecutionSubjectIdentityV3 {
    sha256: [u8; 32],
}
impl InertCompilerExecutionSubjectIdentityV3 {
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
    pub const fn byte_len(self) -> u64 {
        codec::BYTES as u64
    }
    /// Checks fixed wire identity, not payload agreement or execution origin.
    /// Keep the complete borrowed input owner paid on this same account.
    pub fn matches_canonical_bytes(self, bytes: &[u8], budget: &mut Budget<'_>) -> Result<bool> {
        metered(budget, input_floor(bytes), || {
            Ok(bytes.len() == codec::BYTES
                && bytes[codec::SUBJECT_PREIMAGE_BYTES..] == self.sha256
                && SCHEMA.identity(&bytes[..codec::SUBJECT_PREIMAGE_BYTES]) == self.sha256)
        })
    }
}

/// Additional fixed subject/receipt storage, returned UNRESERVED.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InertCompilerExecutionSubjectStorageV3(usize);
impl InertCompilerExecutionSubjectStorageV3 {
    /// Reserve before retaining or using the accompanying subject.
    pub const fn retained_storage(self) -> usize {
        self.0
    }
}

/// Exact V5 conditional capsule/module/transaction content, not original compiler
/// custody, currentness, proof, issuer, publication, load or launch authority.
/// Decoding validates framing only. Compare against reconstruction from the exact
/// typed handoff and independently authenticated occurrence before any admission.
///
/// All seven bindings preserve the V1/V2 field meanings. In particular, the final
/// module field is the lineage RECEIPT identity, not the FFI commitment trailer.
/// The complete V5 capsule binds conditional source/history/catalog/descriptor
/// and target metadata without a second independently maintained projection.
/// Keep input backing/metadata and prior decoder work paid; returned storage is
/// additional and unreserved. No recovery callback runs in the codec scope.
///
/// ```compile_fail
/// use fe2o3_artifact_transaction::InertCompilerExecutionSubjectV3;
/// fn duplicate(v: InertCompilerExecutionSubjectV3) { let _ = v.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_artifact_transaction::{CompilerModuleHandoffReceiptV4, InertCompilerExecutionSubjectV3};
/// use fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1;
/// fn mix(r: CompilerModuleHandoffReceiptV4, h: &InertSemanticCompilerModuleHandoffV5,
///        b: &mut CanonicalKernelIrVerificationResourceBudgetV1<'_>) {
///     let _ = InertCompilerExecutionSubjectV3::from_publication(r, h, b);
/// }
/// ```
/// ```compile_fail
/// use fe2o3_artifact_transaction::{CompilerModuleHandoffReceiptV5, InertCompilerExecutionSubjectV3};
/// use fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV4;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1;
/// fn mix(r: CompilerModuleHandoffReceiptV5, h: &InertSemanticCompilerModuleHandoffV4,
///        b: &mut CanonicalKernelIrVerificationResourceBudgetV1<'_>) {
///     let _ = InertCompilerExecutionSubjectV3::from_publication(r, h, b);
/// }
/// ```
/// ```compile_fail
/// use fe2o3_artifact_transaction::{ConsumedCompilerModuleHandoffV4, InertCompilerExecutionSubjectV3};
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1;
/// fn mix(c: &ConsumedCompilerModuleHandoffV4, b: &mut CanonicalKernelIrVerificationResourceBudgetV1<'_>) {
///     let _ = InertCompilerExecutionSubjectV3::from_consumed(c, b);
/// }
/// ```
/// ```compile_fail
/// use fe2o3_artifact_transaction::{ConsumedCompilerModuleHandoffV5, InertCompilerExecutionSubjectV3};
/// use fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1;
/// fn mapped(c: &ConsumedCompilerModuleHandoffV5<Box<InertSemanticCompilerModuleHandoffV5>>,
///           b: &mut CanonicalKernelIrVerificationResourceBudgetV1<'_>) {
///     let _ = InertCompilerExecutionSubjectV3::from_consumed(c, b);
/// }
/// ```
/// ```compile_fail
/// use fe2o3_artifact_transaction::{InertCompilerExecutionSubjectV2, InertCompilerExecutionSubjectV3};
/// fn downgrade(v: InertCompilerExecutionSubjectV3) -> InertCompilerExecutionSubjectV2 { v.into() }
/// ```
#[derive(Eq, PartialEq)]
pub struct InertCompilerExecutionSubjectV3 {
    fields: codec::Fields,
    encoded: codec::Encoded,
}
impl InertCompilerExecutionSubjectV3 {
    /// Complete fixed work for inert decoding on an original owned account.
    /// This adds a bounded local window, not a new account or authority.
    pub const COMPOSED_DECODE_WORK: usize =
        Budget::STORAGE_WINDOW_WORK_V1 + INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3;
    /// Complete local overlap including the fixed input and window bookkeeping.
    pub const COMPOSED_DECODE_SCRATCH: usize = Budget::STORAGE_WINDOW_SCRATCH_V1
        + codec::BYTES
        + INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3;

    /// Exact inert codec on the same original Owned::with_budget account, with
    /// a fixed local overlap <=256 MiB above its actual entry floor. Larger
    /// external owners remain paid; no total cap, work history or account is
    /// reset. The complete fixed input must already be paid, and is counted
    /// again inside the local window. No caller can select the excluded floor.
    ///
    /// Success returns additional UNRESERVED storage, exactly like decode().
    /// Framing failure refunds only this codec's temporary storage. This is not
    /// terminal source recovery or protected compiler/receipt admission.
    pub fn decode_in_original_account_v3(
        bytes: &[u8],
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        let floor = budget.storage();
        budget.with_additional_storage_window_v1(Self::COMPOSED_DECODE_SCRATCH, |budget| {
            budget.with_prepaid_scope(
                floor.max(input_floor(bytes)),
                8,
                INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3,
                Self::COMPOSED_DECODE_SCRATCH,
                |_| {
                    let (fields, encoded) = SCHEMA.decode(bytes)?;
                    Ok((
                        Self { fields, encoded },
                        InertCompilerExecutionSubjectStorageV3(RETAINED),
                    ))
                },
            )
        })
    }

    /// Reconstructs from the exact V5 receipt and cached immutable V5 axes.
    pub fn from_publication(
        receipt: CompilerModuleHandoffReceiptV5,
        handoff: &Handoff,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        subject(budget, handoff_floor(handoff)?, || {
            Self::from_receipt(receipt, handoff)
        })
    }

    /// Same inert reconstruction on the original owned account. The actual
    /// complete handoff backing/metadata and fixed receipt are counted again
    /// inside an additional <=256 MiB window. This does not establish currentness
    /// or compiler origin; those remain duties of the actual retained caller.
    pub fn from_publication_in_original_account_v3(
        receipt: CompilerModuleHandoffReceiptV5,
        handoff: &Handoff,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        let inputs = handoff_floor(handoff)?
            .checked_add(size_of::<CompilerModuleHandoffReceiptV5>())
            .ok_or(Resource::Arithmetic)?;
        let floor = budget.storage();
        let overlap = inputs
            .checked_add(Budget::STORAGE_WINDOW_SCRATCH_V1)
            .ok_or(Resource::Arithmetic)?;
        budget.with_additional_storage_window_v1(MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5, |b| {
            b.with_prepaid_scope(floor.max(inputs), 0, 0, overlap, |b| {
                Self::from_publication_in_custody(receipt, handoff, b)
            })
        })
    }

    /// Inert logical work for original-account publication reconstruction.
    /// Caller comparison and retention of the returned owner are additional.
    pub const COMPOSED_PUBLICATION_WORK_V3: usize =
        Budget::STORAGE_WINDOW_WORK_V1 + INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3;

    /// Full nested scratch quote over this actual immutable handoff. It does
    /// not reserve storage, validate occurrence currentness or grant authority.
    pub fn composed_publication_storage_v3(handoff: &Handoff) -> Result<usize> {
        handoff_floor(handoff)?
            .checked_add(size_of::<CompilerModuleHandoffReceiptV5>())
            .and_then(|n| n.checked_add(Budget::STORAGE_WINDOW_SCRATCH_V1))
            .and_then(|n| n.checked_add(INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3))
            .ok_or(Resource::Arithmetic.into())
    }

    // Only exact-pair custody composition enters here, under its full-owner
    // local window. The codec and post-entry fees are shared with legacy entry.
    pub(crate) fn from_publication_in_custody(
        receipt: CompilerModuleHandoffReceiptV5,
        handoff: &Handoff,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        let floor = handoff_floor(handoff)?;
        crate::compiler_module_handoff::resources::with_budget(budget, |budget| {
            budget.charge_work(8)?;
            metered_after_entry(budget, floor, || {
                Ok((
                    Self::from_receipt(receipt, handoff)?,
                    InertCompilerExecutionSubjectStorageV3(RETAINED),
                ))
            })
        })?
    }

    /// Concrete raw consumed transport only. For a verified owner use
    /// `from_publication(consumed.receipt(), owner.handoff(), budget)` with all
    /// owner storage prepaid. This fixed codec never calls user-defined `AsRef`.
    pub fn from_consumed(
        consumed: &ConsumedCompilerModuleHandoffV5,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        subject(budget, consumed.storage().retained_storage(), || {
            Self::from_receipt(consumed.receipt(), consumed.handoff())
        })
    }

    fn from_receipt(receipt: CompilerModuleHandoffReceiptV5, handoff: &Handoff) -> Result<Self> {
        if receipt.handoff_identity() != handoff.identity() {
            return Err(Failure::HandoffIdentityMismatch);
        }
        if receipt.length() != handoff.canonical_bytes().len() {
            return Err(Failure::HandoffLengthMismatch);
        }
        Self::from_exact(
            receipt.attempt(),
            receipt.slot(),
            receipt.transaction_identity(),
            handoff,
        )
    }

    /// Binds supplied occurrence coordinates without authenticating execution or
    /// manufacturing a consumed transaction. A coherent alternative stays inert.
    pub fn from_replay_evidence(
        attempt: BuildAttempt,
        slot: CompilerModuleHandoffSlotV5,
        transaction: CompilerModuleHandoffTransactionIdentityV5,
        handoff: &Handoff,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        subject(budget, handoff_floor(handoff)?, || {
            Self::from_exact(attempt, slot, transaction, handoff)
        })
    }

    /// Same inert replay reconstruction on the original owned account. The
    /// complete handoff backing/metadata and fixed occurrence coordinates must
    /// be prepaid and are counted again inside an additional <=256 MiB window.
    /// Returns the full result charge unreserved; legacy replay stays capped.
    ///
    /// This binds no input to a ledger and authenticates neither the occurrence
    /// nor execution. A separately prepaid account may reconstruct the same
    /// inert bytes; protected callers must retain their original owners/account
    /// and independently join the complete subject to authenticated evidence.
    pub fn from_replay_evidence_in_original_account_v3(
        attempt: BuildAttempt,
        slot: CompilerModuleHandoffSlotV5,
        transaction: CompilerModuleHandoffTransactionIdentityV5,
        handoff: &Handoff,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        let inputs = replay_input_floor(handoff)?;
        let floor = budget.storage();
        let overlap = inputs
            .checked_add(Budget::STORAGE_WINDOW_SCRATCH_V1)
            .ok_or(Resource::Arithmetic)?;
        budget.with_additional_storage_window_v1(MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5, |b| {
            b.with_prepaid_scope(floor.max(inputs), 0, 0, overlap, |b| {
                crate::compiler_module_handoff::resources::with_budget(b, |b| {
                    b.charge_work(8)?;
                    metered_after_entry(b, inputs, || {
                        Ok((
                            Self::from_exact(attempt, slot, transaction, handoff)?,
                            InertCompilerExecutionSubjectStorageV3(RETAINED),
                        ))
                    })
                })?
            })
        })
    }

    /// Logical work for original-account replay, excluding caller comparisons.
    pub const COMPOSED_REPLAY_WORK_V3: usize =
        Budget::STORAGE_WINDOW_WORK_V1 + INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3;

    /// Additional nested scratch over the actual handoff and fixed coordinates.
    /// Inputs must already be prepaid; the returned subject charge is separate.
    pub fn composed_replay_storage_v3(handoff: &Handoff) -> Result<usize> {
        replay_input_floor(handoff)?
            .checked_add(Budget::STORAGE_WINDOW_SCRATCH_V1)
            .and_then(|n| n.checked_add(INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3))
            .ok_or(Resource::Arithmetic.into())
    }

    fn from_exact(
        attempt: BuildAttempt,
        slot: CompilerModuleHandoffSlotV5,
        transaction: CompilerModuleHandoffTransactionIdentityV5,
        handoff: &Handoff,
    ) -> Result<Self> {
        let capsule = handoff.capsule();
        let inventory = capsule.rustc_identity_inventory().identity();
        let preflight = capsule.rustc_preflight_plan().identity();
        let commitment = capsule.final_compiler_module_commitment().identity();
        let capsule_identity = capsule.identity();
        let module = handoff.module_handoff().identity();
        let pair = handoff.pair_binding_identity();
        let outer = handoff.identity();
        Self::from_fields(codec::Fields {
            attempt,
            slot: slot as u8,
            transaction_identity: *transaction.as_bytes(),
            rustc_invocation_sha256: capsule.invocation_digest().into_bytes(),
            compiler_closure: *capsule.invocation().compiler_closure(),
            rustc_identity_inventory: Binding::new(
                *inventory.sha256(),
                inventory.byte_len(),
                "rustc identity inventory",
            )?,
            rustc_preflight_plan: Binding::new(
                *preflight.sha256(),
                preflight.byte_len(),
                "rustc preflight plan",
            )?,
            semantic_capsule: Binding::new(
                *capsule_identity.sha256(),
                capsule_identity.byte_len(),
                "semantic capsule",
            )?,
            final_compiler_module_commitment: Binding::new(
                *commitment.sha256(),
                commitment.byte_len(),
                "final compiler module commitment",
            )?,
            compiler_module_handoff: Binding::new(
                *module.sha256(),
                module.byte_len(),
                "compiler module handoff",
            )?,
            compiler_module_pair_binding: Binding::new(
                *pair.sha256(),
                pair.byte_len(),
                "compiler module pair binding",
            )?,
            outer_handoff: Binding::new(*outer.sha256(), outer.byte_len(), "outer V5 handoff")?,
        })
    }

    fn from_fields(fields: codec::Fields) -> Result<Self> {
        let encoded = SCHEMA.encode(&fields)?;
        Ok(Self { fields, encoded })
    }

    /// Strict fixed-wire decode; content lengths never allocate or authenticate.
    /// Invalid physical lengths refuse after fixed prepayment, without traversal.
    pub fn decode(
        bytes: &[u8],
        budget: &mut Budget<'_>,
    ) -> Result<(Self, InertCompilerExecutionSubjectStorageV3)> {
        subject(budget, input_floor(bytes), || {
            let (fields, encoded) = SCHEMA.decode(bytes)?;
            Ok(Self { fields, encoded })
        })
    }

    pub const fn attempt(&self) -> BuildAttempt {
        self.fields.attempt
    }
    pub const fn slot(&self) -> CompilerModuleHandoffSlotV5 {
        CompilerModuleHandoffSlotV5::Production
    }
    pub const fn transaction_identity(&self) -> CompilerModuleHandoffTransactionIdentityV5 {
        CompilerModuleHandoffTransactionIdentityV5::from_bytes(self.fields.transaction_identity)
    }
    pub const fn rustc_invocation_sha256(&self) -> &[u8; 32] {
        &self.fields.rustc_invocation_sha256
    }
    pub const fn compiler_closure(&self) -> CompilerClosureV2 {
        self.fields.compiler_closure
    }
    pub const fn rustc_identity_inventory(&self) -> Binding {
        self.fields.rustc_identity_inventory
    }
    pub const fn rustc_preflight_plan(&self) -> Binding {
        self.fields.rustc_preflight_plan
    }
    pub const fn semantic_capsule(&self) -> Binding {
        self.fields.semantic_capsule
    }
    /// Existing lineage receipt domain, not the FFI commitment's inner identity.
    pub const fn final_compiler_module_commitment(&self) -> Binding {
        self.fields.final_compiler_module_commitment
    }
    pub const fn compiler_module_handoff(&self) -> Binding {
        self.fields.compiler_module_handoff
    }
    pub const fn compiler_module_pair_binding(&self) -> Binding {
        self.fields.compiler_module_pair_binding
    }
    pub const fn outer_handoff(&self) -> Binding {
        self.fields.outer_handoff
    }
    pub const fn identity(&self) -> InertCompilerExecutionSubjectIdentityV3 {
        InertCompilerExecutionSubjectIdentityV3 {
            sha256: self.encoded.sha256,
        }
    }
    pub const fn canonical_bytes(&self) -> &[u8; INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V3] {
        &self.encoded.canonical_bytes
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        false
    }
    pub const fn requires_protected_execution_attestation(&self) -> bool {
        true
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        false
    }
    pub const fn grants_publication_authority(&self) -> bool {
        false
    }
    pub const fn grants_load_authority(&self) -> bool {
        false
    }
    pub const fn grants_launch_authority(&self) -> bool {
        false
    }
}

const _: () = assert!(
    InertCompilerExecutionSubjectV3::COMPOSED_DECODE_SCRATCH
        <= MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5
);
impl fmt::Debug for InertCompilerExecutionSubjectV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InertCompilerExecutionSubjectV3")
            .field("attempt", &self.attempt())
            .field("identity", &self.identity())
            .finish_non_exhaustive()
    }
}

/// Shared framing diagnostics do not invoke a legacy subject decoder.
#[derive(Debug)]
pub enum CompilerExecutionSubjectErrorV3 {
    Framing(CompilerExecutionSubjectErrorV1),
    HandoffIdentityMismatch,
    HandoffLengthMismatch,
    Resource(Resource),
}
type Failure = CompilerExecutionSubjectErrorV3;
type Result<T> = std::result::Result<T, Failure>;
impl From<CompilerExecutionSubjectErrorV1> for Failure {
    fn from(e: CompilerExecutionSubjectErrorV1) -> Self {
        Self::Framing(e)
    }
}
impl From<Resource> for Failure {
    fn from(e: Resource) -> Self {
        Self::Resource(e)
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Framing(e) => e.fmt(f),
            Self::Resource(e) => e.fmt(f),
            Self::HandoffIdentityMismatch => {
                f.write_str("publication binding and conditional V5 identities differ")
            }
            Self::HandoffLengthMismatch => {
                f.write_str("publication binding and conditional V5 lengths differ")
            }
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Framing(e) => Some(e),
            Self::Resource(e) => Some(e),
            _ => None,
        }
    }
}

fn handoff_floor(handoff: &Handoff) -> Result<usize> {
    handoff
        .backing_capacity()
        .checked_add(METADATA)
        .ok_or(Resource::Arithmetic.into())
}
fn input_floor(bytes: &[u8]) -> usize {
    if bytes.len() == codec::BYTES {
        codec::BYTES
    } else {
        0
    }
}
fn replay_input_floor(handoff: &Handoff) -> Result<usize> {
    handoff_floor(handoff)?
        .checked_add(size_of::<(
            BuildAttempt,
            CompilerModuleHandoffSlotV5,
            CompilerModuleHandoffTransactionIdentityV5,
        )>())
        .ok_or(Resource::Arithmetic.into())
}
fn subject(
    budget: &mut Budget<'_>,
    floor: usize,
    f: impl FnOnce() -> Result<InertCompilerExecutionSubjectV3>,
) -> Result<(
    InertCompilerExecutionSubjectV3,
    InertCompilerExecutionSubjectStorageV3,
)> {
    metered(budget, floor, || {
        Ok((f()?, InertCompilerExecutionSubjectStorageV3(RETAINED)))
    })
}
fn metered<T>(budget: &mut Budget<'_>, floor: usize, f: impl FnOnce() -> Result<T>) -> Result<T> {
    crate::compiler_module_handoff::resources::with_budget(budget, |budget| {
        budget.charge_work(8)?;
        if budget.storage_limit() > MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5 {
            return Err(Resource::Accounting.into());
        }
        metered_after_entry(budget, floor, f)
    })?
}

fn metered_after_entry<T>(
    budget: &mut Budget<'_>,
    floor: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if budget.storage() < floor {
        return Err(Resource::Accounting.into());
    }
    budget.reserve_storage(INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3)?;
    budget.charge_work(INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3 - 8)?;
    f()
}

#[cfg(test)]
#[path = "compiler_execution_subject_v3_tests.rs"]
mod tests;
