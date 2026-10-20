//! Independent conditional content recovery, not original compiler custody.
//! Recorded resource denials are terminal here, even if later charges would fit.
#![allow(
    clippy::result_large_err,
    clippy::large_enum_variant,
    reason = "typed terminal errors without uncharged error allocations"
)]

use crate::compiler_native_conditional_source_proof_v2::final_replay::{
    account::{self, Account},
    validate_native_conditional_source_through_f_using_original_account_v2,
    validate_native_conditional_source_through_f_using_v2,
    validate_native_conditional_source_through_f_with_cpu_mapping_using_v2,
    validate_native_conditional_source_through_f_with_cpu_origins_using_original_account_v2,
    validate_native_conditional_source_through_f_with_cpu_origins_using_v2,
};
use crate::{
    NativeConditionalCpuExpectationV1, NativeConditionalFinalErrorV2,
    NativeConditionalFinalInputsV2, NativeConditionalRootPolicyV2,
    NativeConditionalSourceProofErrorV2, ReplayedNativeConditionalSourceV2,
};
use crate::{NativeConditionalCpuMappingContextV1, NativeConditionalCpuMappingExpectationV1};
use fe2o3_amd_target::ProductionAmdTargetProfileV1 as Profile;
use fe2o3_compiler_ffi::{
    INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5 as METADATA,
    InertSemanticCompilerModuleHandoffV5 as Handoff,
};
use fe2o3_compiler_lineage::{
    RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1, read_rustc_enrollment_inventory_v1,
};
use fe2o3_kernel_descriptor::{
    DESCRIPTOR_READER_SCRATCH_STORAGE_V5, DESCRIPTOR_TABLE_VIEW_STORAGE_V5,
    decode_device_descriptor_table_v5,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
    InertCanonicalKernelIrContractCatalogV1 as Catalog,
    VerifiedCanonicalKernelIrModuleV12 as Graph,
};
/// Resource limits accepted by the public conditional handoff recovery routes.
pub use fe2o3_kernel_opt::CanonicalRefinedForwardingHistoryLimitsV1;
use fe2o3_kernel_opt::{
    CanonicalRefinedForwardingHistoryLimitsV1 as Limits,
    MAX_REFINED_FORWARDING_HISTORY_STORAGE_V1 as MAX_STORAGE,
    RefinedForwardingHistoryRoleV1 as Role,
    materialize_refined_forwarding_history_in_original_account_v1,
    materialize_refined_forwarding_history_v1,
    read_refined_forwarding_history_in_original_account_v1, read_refined_forwarding_history_v1,
};
use std::mem::size_of;

const _: () = {
    use fe2o3_compiler_lineage as framing;
    assert!(framing::MAX_NATIVE_CONDITIONAL_STORAGE_V1 == MAX_STORAGE);
    assert!(
        framing::MAX_NATIVE_CONDITIONAL_HISTORY_BYTES_V1
            == fe2o3_kernel_opt::MAX_REFINED_FORWARDING_HISTORY_BYTES_V1
    );
    assert!(
        framing::MAX_NATIVE_CONDITIONAL_CATALOG_BYTES_V1
            == fe2o3_kernel_ir::MAX_KERNEL_IR_CONTRACT_CATALOG_BYTES_V1
    );
    assert!(
        framing::MAX_NATIVE_CONDITIONAL_DESCRIPTOR_BYTES_V1
            == fe2o3_kernel_descriptor::MAX_DESCRIPTOR_TABLE_BYTES
    );
    assert!(
        framing::MAX_NATIVE_CONDITIONAL_SOURCE_BYTES_V1
            == framing::MAX_NATIVE_REFINED_FORWARDING_SOURCE_BYTES_V1
    );
};

#[path = "compiler_native_conditional_handoff_v5/error.rs"]
mod error;
#[path = "compiler_native_conditional_handoff_v5/joins.rs"]
mod joins;
pub use error::CompilerConditionalNativeSemanticHandoffErrorV5;
use error::{Cause, CompilerConditionalNativeSemanticHandoffErrorV5 as Error};

/// Additional retained source/F/catalog and wrapper storage, UNRESERVED on
/// return. The exact transport backing and decoded metadata stay separately paid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoveredCompilerConditionalNativeSemanticHandoffStorageV5(usize);
impl RecoveredCompilerConditionalNativeSemanticHandoffStorageV5 {
    /// Reserve before retaining or using the accompanying owner.
    pub const fn retained_storage(self) -> usize {
        self.0
    }
}

/// Move-only independent conditional source/F/V5 content. This retains the
/// original decoded transport and moves the actual F allocation after all joins
/// and nested account postchecks; no optimizer or second strict import runs.
///
/// Content does not authenticate original rustc/nominal/registration custody,
/// protected execution, currentness, or launch restrictions. Inventory/preflight
/// remain inert commitments. There is no ordinary proof or native publication
/// conversion, and no artifact, machine, host or launch authority.
///
/// ```
/// use fe2o3_verifier::RecoveredCompilerConditionalNativeSemanticHandoffV5 as Owner;
/// use fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5 as Handoff;
/// use fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12 as Graph;
/// fn borrow(owner: &Owner) -> (&Handoff, &Graph) {
///     (owner.as_ref(), owner.output())
/// }
/// ```
///
/// ```compile_fail
/// use fe2o3_verifier::RecoveredCompilerConditionalNativeSemanticHandoffV5 as Owner;
/// fn duplicate(value: Owner) { let _ = value.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::RecoveredCompilerConditionalNativeSemanticHandoffV5 as Owner;
/// fn manufacture() -> Owner { Owner::default() }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::{RecoveredCompilerConditionalNativeSemanticHandoffV5 as Conditional,
///     RecoveredCompilerNativeSemanticHandoffV4 as Ordinary};
/// fn downgrade(value: Conditional) -> Ordinary { value.into() }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::RecoveredCompilerConditionalNativeSemanticHandoffV5 as Owner;
/// fn mutate(value: Owner) { value.output().module().kernels.clear(); }
/// ```
#[must_use = "reserve the returned additional storage before using this owner"]
pub struct RecoveredCompilerConditionalNativeSemanticHandoffV5 {
    handoff: Handoff,
    parts: Parts,
    storage: RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
}
struct Parts {
    source: ReplayedNativeConditionalSourceV2,
    output: Graph,
    catalog: Catalog,
    limits: Limits,
    profile: Profile,
}
impl AsRef<Handoff> for RecoveredCompilerConditionalNativeSemanticHandoffV5 {
    fn as_ref(&self) -> &Handoff {
        &self.handoff
    }
}
impl RecoveredCompilerConditionalNativeSemanticHandoffV5 {
    /// Exact decoded V5 transport, including the unchanged module V2.
    pub const fn handoff(&self) -> &Handoff {
        &self.handoff
    }
    /// Reconstructed conditional source content, not original producer custody.
    pub const fn source(&self) -> &ReplayedNativeConditionalSourceV2 {
        &self.parts.source
    }
    /// Actual decoded final graph allocation, borrowed immutably.
    pub const fn output(&self) -> &Graph {
        &self.parts.output
    }
    /// Exact decoded final contract catalog.
    pub const fn catalog(&self) -> &Catalog {
        &self.parts.catalog
    }
    /// Independently accepted complete history limits.
    pub const fn limits(&self) -> Limits {
        self.parts.limits
    }
    /// Independently accepted target profile.
    pub const fn profile(&self) -> Profile {
        self.parts.profile
    }
    /// Additional unreserved custody extent, excluding prepaid transport.
    pub const fn storage(&self) -> RecoveredCompilerConditionalNativeSemanticHandoffStorageV5 {
        self.storage
    }
}

type Output = (
    RecoveredCompilerConditionalNativeSemanticHandoffV5,
    RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
);
// Conservative wrapper headers coexist with the individual engine receipts.
// The outer Account remains live inside the five paid final-composition scopes.
const HEADER: usize = size_of::<RecoveredCompilerConditionalNativeSemanticHandoffV5>();
const WORKING: usize = size_of::<Account>()
    + size_of::<Result<(Parts, usize), Error>>()
    + size_of::<Result<Output, Error>>()
    + size_of::<[usize; 8]>();

type CpuOrigins<'a> = Option<&'a [NativeConditionalCpuExpectationV1]>;

#[derive(Clone, Copy)]
enum CpuSelection<'a> {
    Legacy(CpuOrigins<'a>),
    Mapping(&'a NativeConditionalCpuMappingExpectationV1),
}
impl<'a> From<CpuOrigins<'a>> for CpuSelection<'a> {
    fn from(value: CpuOrigins<'a>) -> Self {
        Self::Legacy(value)
    }
}
fn selection_backing(selection: CpuSelection<'_>) -> usize {
    match selection {
        CpuSelection::Legacy(origins) => origin_backing(origins),
        CpuSelection::Mapping(_) => size_of::<NativeConditionalCpuMappingExpectationV1>(),
    }
}
fn selection_working(selection: CpuSelection<'_>) -> usize {
    match selection {
        CpuSelection::Legacy(origins) => origin_working(origins),
        CpuSelection::Mapping(_) => {
            RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1
                + size_of::<NativeConditionalCpuMappingContextV1<'_>>()
                + size_of::<CpuSelection<'_>>()
                + 2 * size_of::<usize>()
        }
    }
}

fn origin_backing(origins: CpuOrigins<'_>) -> usize {
    origins.map_or(0, std::mem::size_of_val)
}

fn origin_working(origins: CpuOrigins<'_>) -> usize {
    if origins.is_some() {
        size_of::<CpuOrigins<'_>>() + size_of::<usize>()
    } else {
        0
    }
}

fn sum(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(0usize, |n, v| n.checked_add(*v))
        .ok_or_else(|| Resource::Arithmetic.into())
}

/// Recovers conditional source/F/V5 content with independently accepted ordered
/// policies, complete history limits and target profile. Never derives policy
/// keys or acceptance limits from transport. Each root uses the existing single
/// decoded CPU scope and strict formula import; the final text relation is joined
/// within that same composition, not replayed by a second consumer.
///
/// Caller must prepay the entire `backing_capacity()` (including spare/enclosing
/// bytes), V5 decode metadata storage and the published decode work quote before
/// decoding the handoff. Success returns additional storage UNRESERVED. Errors
/// and unwinds retain terminal reservations; NEVER enclose this entry in an
/// ordinary blanket-refund scope. Transaction currentness/lock and original
/// producer custody remain mandatory external duties. No finalizer gate changes.
pub fn recover_compiler_conditional_native_semantic_handoff_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    let entry = begin(handoff.backing_capacity(), budget)?;
    recover_using(
        handoff,
        accepted,
        CpuSelection::Legacy(None),
        expected_limits,
        profile,
        &entry,
        false,
        budget,
    )
}

/// Same terminal V5 recovery on the original owned account. The complete actual
/// handoff backing, metadata and borrowed policy backing are counted AGAIN under
/// an additional <=256 MiB ceiling. Nested history operations use their concrete
/// original-account variants. No new budget, cap increase or policy authority.
///
/// Additional entry work is STORAGE_WINDOW_WORK_V1 + 8 + accepted.len(); each
/// of the three nested history phases additionally charges its documented
/// composition overhead. Errors/unwinds preserve all partial reservations. The
/// returned successful owner/charge remains additional and unreserved.
pub fn recover_compiler_conditional_native_semantic_handoff_in_original_account_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    budget.check_prior_denials_v1()?;
    let policy = crate::compiler_native_conditional_policy_roster_v1::native_conditional_root_policy_input_storage_v2(
        accepted, budget,
    )?;
    let inputs = sum(&[handoff.backing_capacity(), METADATA, policy])?;
    original_recovery(inputs, budget, |b| {
        let entry = begin_bounded(handoff.backing_capacity(), b)?;
        recover_using(
            handoff,
            accepted,
            CpuSelection::Legacy(None),
            expected_limits,
            profile,
            &entry,
            true,
            b,
        )
    })
}

/// Explicit per-root CPU codec selection; existing recovery remains V1-only.
/// The independently retained expectation roster and full handoff/metadata
/// backing must be prepaid together. Expectations are inert, not Loan custody.
/// This returns the same content-only owner and preserves terminal failures.
pub fn recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    let origins = Some(expected_cpu);
    let entry = begin_with_origins(handoff.backing_capacity(), expected_cpu, budget)?;
    recover_using(
        handoff,
        accepted,
        origins.into(),
        expected_limits,
        profile,
        &entry,
        false,
        budget,
    )
}

/// Origin-selected recovery in the same original-account storage window.
/// Complete handoff, metadata, borrowed policy and expectation backing count
/// again under the existing additional <=256 MiB ceiling. No new account or
/// enrollment authority is created; errors and unwinds retain reservations.
pub fn recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    budget.check_prior_denials_v1()?;
    let origins = Some(expected_cpu);
    let policy = crate::compiler_native_conditional_policy_roster_v1::native_conditional_root_policy_input_storage_v2(
        accepted, budget,
    )?;
    let inputs = sum(&[
        handoff.backing_capacity(),
        METADATA,
        policy,
        origin_backing(origins),
    ])?;
    original_recovery(inputs, budget, |b| {
        let entry = begin_bounded_with_origins(handoff.backing_capacity(), expected_cpu, b)?;
        recover_using(
            handoff,
            accepted,
            origins.into(),
            expected_limits,
            profile,
            &entry,
            true,
            b,
        )
    })
}

fn original_recovery<T>(
    inputs: usize,
    budget: &mut Budget<'_>,
    operation: impl FnOnce(&mut Budget<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    budget.check_prior_denials_v1()?;
    let overlap = sum(&[inputs, Budget::STORAGE_WINDOW_SCRATCH_V1])?;
    let floor = budget.storage();
    budget.with_additional_storage_window_v1(MAX_STORAGE, |b| {
        if floor < inputs {
            return Err(Resource::Accounting.into());
        }
        b.reserve_storage(overlap)?;
        let result = operation(b)?;
        if b.storage() != floor.checked_add(overlap).ok_or(Resource::Arithmetic)? {
            return Err(Resource::Accounting.into());
        }
        b.check_prior_denials_v1()?;
        b.release_storage(overlap)?;
        Ok(result)
    })
}

fn recover_using(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: CpuSelection<'_>,
    expected_limits: Limits,
    profile: Profile,
    entry: &Account,
    original: bool,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    let result = replay(
        &handoff,
        accepted,
        origins,
        expected_limits,
        profile,
        entry,
        original,
        budget,
    );
    let (parts, retained) = if matches!(origins, CpuSelection::Legacy(None)) {
        finish(entry, budget, result)
    } else {
        finish_with_origin_working(entry, selection_working(origins), budget, result)
    }?;
    let storage = RecoveredCompilerConditionalNativeSemanticHandoffStorageV5(retained);
    Ok((
        RecoveredCompilerConditionalNativeSemanticHandoffV5 {
            handoff,
            parts,
            storage,
        },
        storage,
    ))
}

fn begin(backing_capacity: usize, budget: &mut Budget<'_>) -> Result<Account, Error> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(8)?;
    if budget.storage_limit() > MAX_STORAGE {
        return Err(Error::mismatch("bounded conditional native storage cap"));
    }
    begin_after_entry(backing_capacity, budget)
}

fn begin_bounded(backing_capacity: usize, budget: &mut Budget<'_>) -> Result<Account, Error> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(8)?;
    begin_after_entry(backing_capacity, budget)
}

fn begin_with_origins(
    backing_capacity: usize,
    expected: &[NativeConditionalCpuExpectationV1],
    budget: &mut Budget<'_>,
) -> Result<Account, Error> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(10)?;
    if budget.storage_limit() > MAX_STORAGE {
        return Err(Error::mismatch("bounded conditional native storage cap"));
    }
    begin_after_entry_with_origins(backing_capacity, Some(expected), budget)
}

fn begin_bounded_with_origins(
    backing_capacity: usize,
    expected: &[NativeConditionalCpuExpectationV1],
    budget: &mut Budget<'_>,
) -> Result<Account, Error> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(10)?;
    begin_after_entry_with_origins(backing_capacity, Some(expected), budget)
}

fn begin_after_entry(backing_capacity: usize, budget: &mut Budget<'_>) -> Result<Account, Error> {
    begin_after_entry_with_origins(backing_capacity, None, budget)
}

fn begin_after_entry_with_origins(
    backing_capacity: usize,
    origins: CpuOrigins<'_>,
    budget: &mut Budget<'_>,
) -> Result<Account, Error> {
    begin_selected(backing_capacity, origins.into(), budget)
}

fn begin_selected(
    backing_capacity: usize,
    origins: CpuSelection<'_>,
    budget: &mut Budget<'_>,
) -> Result<Account, Error> {
    budget.check_prior_denials_v1()?;
    let entry = Account::capture(budget);
    if budget.storage() < sum(&[backing_capacity, METADATA, selection_backing(origins)])? {
        return Err(Resource::Accounting.into());
    }
    budget.reserve_storage(sum(&[HEADER, WORKING, selection_working(origins)])?)?;
    Ok(entry)
}

// Private concrete-route transfer; caller pays its result/header in WORKING.
fn finish<T>(
    entry: &Account,
    budget: &mut Budget<'_>,
    result: Result<(T, usize), Error>,
) -> Result<(T, usize), Error> {
    finish_with_origin_working(entry, 0, budget, result)
}

fn finish_with_origin_working<T>(
    entry: &Account,
    origin_working: usize,
    budget: &mut Budget<'_>,
    result: Result<(T, usize), Error>,
) -> Result<(T, usize), Error> {
    let working = sum(&[WORKING, origin_working])?;
    if let Err(error) = entry.require(budget, sum(&[HEADER, working])?) {
        drop(result);
        return Err(error.into());
    }
    budget.check_prior_denials_v1()?;
    let (parts, retained) = result?;
    if let Err(error) = entry.require_exact(budget, sum(&[retained, working])?) {
        drop(parts);
        return Err(error.into());
    }
    // This known success-only transfer is the sole unreservation of live custody.
    // All retired temporary owners dropped inside replay before their releases.
    budget.release_storage(sum(&[retained, working])?)?;
    Ok((parts, retained))
}

fn replay(
    handoff: &Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: CpuSelection<'_>,
    expected_limits: Limits,
    profile: Profile,
    entry: &Account,
    original: bool,
    budget: &mut Budget<'_>,
) -> Result<(Parts, usize), Error> {
    let capsule = handoff.capsule();
    let frame = if original {
        read_refined_forwarding_history_in_original_account_v1(capsule.history_bytes(), budget)
    } else {
        read_refined_forwarding_history_v1(capsule.history_bytes(), budget)
    }
    .map_err(|e| Error(Cause::History(e)))?;
    let frame_storage = frame.storage().retained_storage();
    budget.reserve_storage(frame_storage)?;
    let history = if original {
        materialize_refined_forwarding_history_in_original_account_v1(&frame, budget)
    } else {
        materialize_refined_forwarding_history_v1(&frame, budget)
    }
    .map_err(|e| Error(Cause::History(e)))?;
    let history_storage = history.storage().retained_storage();
    budget.reserve_storage(history_storage)?;
    let (catalog, receipt) = Catalog::decode_with_budget(capsule.catalog_bytes(), budget)
        .map_err(|e| Error(Cause::Catalog(e)))?;
    let catalog_storage = receipt.retained_storage();
    budget.reserve_storage(catalog_storage)?;
    budget
        .reserve_storage(DESCRIPTOR_TABLE_VIEW_STORAGE_V5 + DESCRIPTOR_READER_SCRATCH_STORAGE_V5)?;
    let table = decode_device_descriptor_table_v5(capsule.descriptor_bytes(), &mut |n| {
        budget.charge_work(n)
    })
    .map_err(|e| Error(Cause::Descriptor(e)))?;
    budget.release_storage(DESCRIPTOR_READER_SCRATCH_STORAGE_V5)?;
    budget.charge_work(handoff.module_handoff().module_bytes().len())?;
    let final_llvm = std::str::from_utf8(handoff.module_handoff().module_bytes())
        .map_err(|e| Error(Cause::Utf8(e)))?;
    let inputs = NativeConditionalFinalInputsV2 {
        decoded_history: &history,
        expected_limits,
        final_catalog: &catalog,
        published_output_bytes: history.graph(Role::F).canonical().canonical_bytes(),
        profile,
        descriptors: &table,
        final_llvm,
    };
    let (source, receipt) = if let CpuSelection::Mapping(expected) = origins {
        // Decode the actual retained signed-metadata field, never an alternate
        // external view with merely equal invocation/policy header claims.
        let inventory = read_rustc_enrollment_inventory_v1(
            capsule.rustc_identity_inventory().canonical_preimage(),
            MAX_STORAGE,
            |work| budget.charge_work(work),
        )
        .map_err(|error| Error(Cause::Inventory(error)))?;
        let mapping = NativeConditionalCpuMappingContextV1 {
            inventory: &inventory,
            expected,
        };
        let result = validate_native_conditional_source_through_f_with_cpu_mapping_using_v2(
            capsule.source_packet_bytes(),
            accepted,
            &mapping,
            inputs,
            budget,
            original,
            |source, _, relation, budget| joins::check(handoff, source, relation, budget),
        );
        drop(inventory);
        result
    } else if let CpuSelection::Legacy(Some(expected_cpu)) = origins {
        if original {
            validate_native_conditional_source_through_f_with_cpu_origins_using_original_account_v2(
                capsule.source_packet_bytes(),
                accepted,
                expected_cpu,
                inputs,
                budget,
                |source, _, relation, budget| joins::check(handoff, source, relation, budget),
            )
        } else {
            validate_native_conditional_source_through_f_with_cpu_origins_using_v2(
                capsule.source_packet_bytes(),
                accepted,
                expected_cpu,
                inputs,
                budget,
                |source, _, relation, budget| joins::check(handoff, source, relation, budget),
            )
        }
    } else if original {
        validate_native_conditional_source_through_f_using_original_account_v2(
            capsule.source_packet_bytes(),
            accepted,
            inputs,
            budget,
            |source, _, relation, budget| joins::check(handoff, source, relation, budget),
        )
    } else {
        validate_native_conditional_source_through_f_using_v2(
            capsule.source_packet_bytes(),
            accepted,
            inputs,
            budget,
            |source, _, relation, budget| joins::check(handoff, source, relation, budget),
        )
    }?;
    let source_storage = receipt.retained_storage();
    budget.reserve_storage(source_storage)?;
    // No graph moves until same-visit joins and all enclosing CPU/source/F
    // postchecks completed, on this exact original ledger and complete floor.
    entry.require_exact(
        budget,
        sum(&[
            HEADER,
            WORKING,
            selection_working(origins),
            frame_storage,
            history_storage,
            catalog_storage,
            DESCRIPTOR_TABLE_VIEW_STORAGE_V5,
            source_storage,
        ])?,
    )?;
    drop(table);
    budget.release_storage(DESCRIPTOR_TABLE_VIEW_STORAGE_V5)?;
    let (output, output_storage) = history.into_final_graph();
    budget.release_storage(
        history_storage
            .checked_sub(output_storage.retained_storage())
            .ok_or(Resource::Accounting)?,
    )?;
    drop(frame);
    budget.release_storage(frame_storage)?;
    let retained = sum(&[
        HEADER,
        source_storage,
        output_storage.retained_storage(),
        catalog_storage,
    ])?;
    Ok((
        Parts {
            source,
            output,
            catalog,
            limits: expected_limits,
            profile,
        },
        retained,
    ))
}

#[cfg(test)]
#[path = "compiler_native_conditional_handoff_v5/tests.rs"]
mod tests;

/// Explicit mapped recovery. Independent coordinates select the new inventory
/// route; malformed or legacy inventory never falls back. Only the actual
/// retained capsule inventory is decoded. This remains content recovery, not
/// installed-policy, authenticated-carriage, original-source or currentness authority.
/// Prepay handoff capacity/metadata and the expectation on the original account.
///
/// ```compile_fail
/// use fe2o3_compiler_lineage::RustcEnrollmentInventoryRefV1;
/// use fe2o3_verifier::*;
/// fn substitute(policy: &[u8], handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
///     alternate: &RustcEnrollmentInventoryRefV1<'_>, budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>) {
///     recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1(policy, handoff, alternate, budget);
/// }
/// ```
pub fn recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected: &NativeConditionalCpuMappingExpectationV1,
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(10)?;
    if budget.storage_limit() > MAX_STORAGE {
        return Err(Error::mismatch("bounded conditional native storage cap"));
    }
    let selection = CpuSelection::Mapping(expected);
    let entry = begin_selected(handoff.backing_capacity(), selection, budget)?;
    recover_using(
        handoff,
        accepted,
        selection,
        expected_limits,
        profile,
        &entry,
        false,
        budget,
    )
}

/// The mapped route with the existing original-account overlap/window protocol.
/// Complete actual input capacities count again; all failures remain terminal.
pub fn recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_in_original_account_v5(
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected: &NativeConditionalCpuMappingExpectationV1,
    expected_limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    budget.check_prior_denials_v1()?;
    let selection = CpuSelection::Mapping(expected);
    let policy = crate::compiler_native_conditional_policy_roster_v1::native_conditional_root_policy_input_storage_v2(accepted, budget)?;
    let inputs = sum(&[
        handoff.backing_capacity(),
        METADATA,
        policy,
        selection_backing(selection),
    ])?;
    original_recovery(inputs, budget, |b| {
        b.charge_work(10)?;
        let entry = begin_selected(handoff.backing_capacity(), selection, b)?;
        recover_using(
            handoff,
            accepted,
            selection,
            expected_limits,
            profile,
            &entry,
            true,
            b,
        )
    })
}
