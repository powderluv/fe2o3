//! CPU-content-bound conditional statements; never source-origin or native authority.
use super::*;
use crate::conditional_reference_v1::{
    ConditionalReferenceErrorV1, ConditionalReferenceInputV1,
    with_source_bound_cpu_correspondence_v1,
};
use crate::functional_refinement_receipt_v2::{
    InertFunctionalRefinementReceiptSignatureV2, import_and_retain_functional_refinement_receipt_v2,
};
use crate::portable_reference_v1::{
    ReferenceReplayInputV1,
    codec::{
        DecodedNativeCpuInputV1, DecodedNativeCpuPolicyInputV2, NativeCpuCodecErrorV1,
        NativeCpuInputV1, NativeCpuPolicyInputV2, with_encoded_native_cpu_input_v1,
        with_encoded_native_cpu_policy_input_v2,
    },
};

type Request<'a> = ProductionSourceBoundConditionalAggregateRequestV1<'a>;
type Error = ProductionConditionalFormulaErrorV2;
const DOMAIN: &[u8] = b"FE2O3/CONDITIONAL-MEMORY-TRANSITION/V2/LE/SHARED-IEEE/CPU-SOURCE\0";
const RETAINED_STORAGE: usize = std::mem::size_of::<RetainedProductionConditionalFormulaV2>();

/// Version-specific refusal; none of these errors carries proof authority.
#[derive(Debug)]
pub enum ProductionConditionalFormulaErrorV2 {
    Formula(ProductionConditionalFormulaErrorV1),
    Codec(NativeCpuCodecErrorV1),
    Correspondence(ConditionalReferenceErrorV1),
    Subject(&'static str),
}
impl From<ProductionConditionalFormulaErrorV1> for Error {
    fn from(value: ProductionConditionalFormulaErrorV1) -> Self {
        Self::Formula(value)
    }
}
impl From<Resource> for Error {
    fn from(value: Resource) -> Self {
        Self::Formula(value.into())
    }
}
impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(out, "conditional CPU-bound formula V2: {self:?}")
    }
}
impl std::error::Error for Error {}

/// Copyable diagnostic identities, not a proof or a V1 report conversion.
///
/// ```compile_fail
/// use fe2o3_verifier::{ProductionConditionalFormulaReportV2, RetainedProductionConditionalFormulaV2};
/// fn install(report: ProductionConditionalFormulaReportV2) -> RetainedProductionConditionalFormulaV2 { report.into() }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionConditionalFormulaReportV2 {
    formula: ProductionConditionalFormulaReportV1,
    cpu_input: DigestV1,
}
impl ProductionConditionalFormulaReportV2 {
    pub const fn statement_identity(self) -> DigestV1 {
        self.formula.statement_identity()
    }
    pub const fn generated_source_identity(self) -> DigestV1 {
        self.formula.generated_source_identity()
    }
    pub const fn binding(self) -> FunctionalRefinementBindingV2 {
        self.formula.binding()
    }
    pub const fn execution_identity(self) -> DigestV1 {
        self.formula.execution_identity()
    }
    pub const fn receipt_identity(self) -> DigestV1 {
        self.formula.receipt_identity()
    }
    pub const fn cpu_input_commitment(self) -> DigestV1 {
        self.cpu_input
    }
}

/// Lent only after fresh CPU/source/statement replay and strict receipt import.
pub struct ProductionConditionalFormulaExecutionV2 {
    report: ProductionConditionalFormulaReportV2,
    retained: RetainedImportedFunctionalRefinementReceiptV2,
}
impl ProductionConditionalFormulaExecutionV2 {
    pub const fn report(&self) -> ProductionConditionalFormulaReportV2 {
        self.report
    }
    pub const fn signed_receipt_wire(&self) -> &[u8] {
        self.retained.wire()
    }
    pub const fn receipt_verifying_key(&self) -> &[u8; 32] {
        self.retained.verifying_key()
    }
}

/// Move-only imported receipt custody for one exact source/graph and CPU input.
/// The caller retains its original accounting phase and source authenticity.
/// Dropping this owner does not release that phase's storage reservation.
///
/// ```compile_fail
/// use fe2o3_verifier::RetainedProductionConditionalFormulaV2;
/// fn duplicate(p: RetainedProductionConditionalFormulaV2) { let _ = p.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::{RetainedProductionConditionalFormulaV1, RetainedProductionConditionalFormulaV2};
/// fn upgrade(p: RetainedProductionConditionalFormulaV1) -> RetainedProductionConditionalFormulaV2 { p.into() }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::{RetainedProductionConditionalFormulaV1, RetainedProductionConditionalFormulaV2};
/// fn downgrade(p: RetainedProductionConditionalFormulaV2) -> RetainedProductionConditionalFormulaV1 { p.into() }
/// ```
/// ```compile_fail
/// use fe2o3_verifier::RetainedProductionConditionalFormulaV2;
/// use fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1;
/// fn promote(p: RetainedProductionConditionalFormulaV2) -> ProductionFormalMemoryOwnerV1 { p.into() }
/// ```
#[must_use = "retain on the original account or drop before releasing its charge"]
pub struct RetainedProductionConditionalFormulaV2 {
    execution: ProductionConditionalFormulaExecutionV2,
    policy: FunctionalRefinementImportPolicyV2,
    subject: retention::RetainedSubjectV1,
}

/// Executes a fresh V2 statement over the same borrowed input encoded and joined.
/// No raw commitment, generated proof text or V1 execution is accepted.
pub fn execute_and_retain_conditional_ranked_formula_v2(
    runtime: &FunctionalRefinementVerusRuntimeLeaseV1,
    request: &Request<'_>,
    input: NativeCpuInputV1<'_>,
    budget: &mut Budget<'_>,
    timeout_seconds: u32,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    execute_and_retain_using(
        runtime,
        request,
        LiveCpuInput::Registration(input),
        budget,
        timeout_seconds,
    )
}

/// Executes a CPU-content-bound statement using the policy-origin V2 codec.
/// The origin description does not authenticate enrollment: the caller must
/// preserve the original invocation, admitted policy and live source owners.
/// This grants no native launch authority and accepts no raw commitment.
///
/// ```compile_fail
/// use fe2o3_verifier::{execute_and_retain_conditional_ranked_formula_policy_v2 as execute, FunctionalRefinementVerusRuntimeLeaseV1};
/// use fe2o3_verifier::portable_reference_v1::codec::NativeCpuInputV1;
/// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
/// fn wrong_origin(runtime: &FunctionalRefinementVerusRuntimeLeaseV1, request: &Request<'_>,
///     input: NativeCpuInputV1<'_>, budget: &mut Budget<'_>) {
///     let _ = execute(runtime, request, input, budget, 1);
/// }
/// ```
pub fn execute_and_retain_conditional_ranked_formula_policy_v2(
    runtime: &FunctionalRefinementVerusRuntimeLeaseV1,
    request: &Request<'_>,
    input: NativeCpuPolicyInputV2<'_>,
    budget: &mut Budget<'_>,
    timeout_seconds: u32,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    execute_and_retain_using(
        runtime,
        request,
        LiveCpuInput::Policy(input),
        budget,
        timeout_seconds,
    )
}

enum LiveCpuInput<'a> {
    Registration(NativeCpuInputV1<'a>),
    Policy(NativeCpuPolicyInputV2<'a>),
}

fn execute_and_retain_using(
    runtime: &FunctionalRefinementVerusRuntimeLeaseV1,
    request: &Request<'_>,
    input: LiveCpuInput<'_>,
    budget: &mut Budget<'_>,
    timeout_seconds: u32,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    retention::retain_reservation_using(budget, RETAINED_STORAGE, |budget| {
        with_live_cpu_input(request, input, budget, |request, cpu_input, budget| {
            with_scratch_using(budget, retention::PREPARATION_STORAGE, |budget| {
                let prepared = prepare_v2(request, cpu_input, budget)?;
                let (formula, retained, policy) =
                    execute_prepared(runtime, request, prepared, budget, timeout_seconds)?;
                require_policy(request, &policy, budget)?;
                Ok(retain(request, formula, cpu_input, retained, policy))
            })
        })
    })
}

/// Reprepares from this same decoded CPU owner and imports actual signed bytes.
/// Accepted policy is external; the signature's embedded key cannot select it.
/// This runs no protected execution and constructs no source-origin authority.
/// The enclosing B1 decode/lower callbacks must also finish before installation.
///
/// ```compile_fail
/// use fe2o3_verifier::{import_and_retain_conditional_ranked_formula_v2 as import, InertFunctionalRefinementReceiptSignatureV2 as Signature};
/// use fe2o3_functional_proof::FunctionalRefinementImportPolicyV2 as Policy;
/// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
/// fn raw_digest(r: &Request<'_>, hash: &[u8; 32], s: &Signature, p: &Policy, b: &mut Budget<'_>) {
///     let _ = import(r, hash, s, p, b);
/// }
/// ```
pub fn import_and_retain_conditional_ranked_formula_v2(
    request: &Request<'_>,
    input: &DecodedNativeCpuInputV1,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    import_and_retain_using(
        request,
        DecodedCpuInput::Registration(input),
        signature,
        accepted,
        budget,
        |owner, _, _| Ok(owner),
    )
}

/// Imports actual signed bytes using this same decoded policy-origin owner.
/// Neither the embedded receipt key nor the inert origin selects accepted
/// authority. Original policy/source custody and outer postchecks stay external.
///
/// ```compile_fail
/// use fe2o3_verifier::{import_and_retain_conditional_ranked_formula_policy_v2 as import, InertFunctionalRefinementReceiptSignatureV2 as Signature};
/// use fe2o3_verifier::portable_reference_v1::codec::DecodedNativeCpuInputV1;
/// use fe2o3_functional_proof::FunctionalRefinementImportPolicyV2 as Policy;
/// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
/// fn wrong_origin(r: &Request<'_>, input: &DecodedNativeCpuInputV1,
///     signature: &Signature, policy: &Policy, budget: &mut Budget<'_>) {
///     let _ = import(r, input, signature, policy, budget);
/// }
/// ```
pub fn import_and_retain_conditional_ranked_formula_policy_v2(
    request: &Request<'_>,
    input: &DecodedNativeCpuPolicyInputV2,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    import_and_retain_using(
        request,
        DecodedCpuInput::Policy(input),
        signature,
        accepted,
        budget,
        |owner, _, _| Ok(owner),
    )
}

/// An opaque refusal of the opt-in check route, not a refundable import error.
/// Enclosing consumers must preserve terminal charges even when a later graph
/// or account postcheck replaced the callback's original error.
#[derive(Debug)]
pub(crate) struct ConditionalFormulaImportCheckErrorV2(Error);

impl fmt::Display for ConditionalFormulaImportCheckErrorV2 {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, out)
    }
}
// Intentionally no Error::source: generic source-chain refund classifiers must
// not turn this route's opaque callback refusal into a refundable formula error.
impl std::error::Error for ConditionalFormulaImportCheckErrorV2 {}

/// Lends the actual execution during the same strict import, never a reimport.
/// Success cannot retain callback storage or a borrowed execution. Callback Err
/// remains nested through graph/B2/importer checks, then destroys the provisional
/// owner without refunding its reservation. All errors of this opt-in route are
/// terminal for enclosing accounting; do not map them to a refundable Subject.
/// Lower/B1/C0 postchecks still must finish before a caller installs the owner.
#[allow(dead_code, reason = "private same-visit C1 continuation prerequisite")]
pub(crate) fn import_and_check_conditional_ranked_formula_v2<E>(
    request: &Request<'_>,
    input: &DecodedNativeCpuInputV1,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
    check: impl for<'proof> FnOnce(
        &'proof ProductionConditionalFormulaExecutionV2,
        &mut Budget<'_>,
    ) -> Result<(), E>,
) -> Result<Result<RetainedProductionConditionalFormulaV2, E>, ConditionalFormulaImportCheckErrorV2>
{
    import_and_check_using(
        request,
        DecodedCpuInput::Registration(input),
        signature,
        accepted,
        budget,
        check,
    )
}

/// The policy decoder uses the identical nonrefundable same-visit continuation.
#[allow(
    dead_code,
    reason = "private policy source/final recovery prerequisite"
)]
pub(crate) fn import_and_check_conditional_ranked_formula_policy_v2<E>(
    request: &Request<'_>,
    input: &DecodedNativeCpuPolicyInputV2,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
    check: impl for<'proof> FnOnce(
        &'proof ProductionConditionalFormulaExecutionV2,
        &mut Budget<'_>,
    ) -> Result<(), E>,
) -> Result<Result<RetainedProductionConditionalFormulaV2, E>, ConditionalFormulaImportCheckErrorV2>
{
    import_and_check_using(
        request,
        DecodedCpuInput::Policy(input),
        signature,
        accepted,
        budget,
        check,
    )
}

fn import_and_check_using<E>(
    request: &Request<'_>,
    input: DecodedCpuInput<'_>,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
    check: impl for<'proof> FnOnce(
        &'proof ProductionConditionalFormulaExecutionV2,
        &mut Budget<'_>,
    ) -> Result<(), E>,
) -> Result<Result<RetainedProductionConditionalFormulaV2, E>, ConditionalFormulaImportCheckErrorV2>
{
    let result = with_scratch_using(budget, IMPORT_CHECK_STORAGE, |budget| {
        import_and_retain_using(
            request,
            input,
            signature,
            accepted,
            budget,
            |owner, request, budget| {
                check_imported_owner(
                    owner,
                    budget,
                    |owner, budget| check(&owner.execution, budget),
                    |budget| current(request.pliron_input(), budget).map_err(Error::from),
                )
            },
        )
    });
    finish_import_check(result)
}

fn finish_import_check<T, E>(
    result: Result<(T, Result<(), E>), Error>,
) -> Result<Result<T, E>, ConditionalFormulaImportCheckErrorV2> {
    let (owner, checked) = result.map_err(ConditionalFormulaImportCheckErrorV2)?;
    Ok(match checked {
        Ok(()) => Ok(owner),
        Err(error) => {
            drop(owner);
            Err(error)
        }
    })
}

struct ImportCheckAccountV2 {
    address: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    floor: usize,
}
const IMPORT_CHECK_STORAGE: usize = std::mem::size_of::<ImportCheckAccountV2>();

impl ImportCheckAccountV2 {
    fn require(&self, budget: &Budget<'_>, exact: bool) -> Result<(), Error> {
        if budget as *const Budget<'_> as usize != self.address
            || budget.work_ledger_identity_v1() != self.ledger
            || budget.storage() < self.floor
            || (exact && budget.storage() != self.floor)
        {
            return Err(Resource::Accounting.into());
        }
        Ok(())
    }
}

// Generic only for inert component witnesses; production supplies the actual
// retained owner from the shared strict importer. This constructs no receipt.
fn check_imported_owner<T, E>(
    owner: T,
    budget: &mut Budget<'_>,
    check: impl for<'owner> FnOnce(&'owner T, &mut Budget<'_>) -> Result<(), E>,
    postcheck: impl FnOnce(&mut Budget<'_>) -> Result<(), Error>,
) -> Result<(T, Result<(), E>), Error> {
    budget.charge_work(9)?;
    let account = ImportCheckAccountV2 {
        address: budget as *const Budget<'_> as usize,
        ledger: budget.work_ledger_identity_v1(),
        floor: budget.storage(),
    };
    let checked = check(&owner, budget);
    account.require(budget, checked.is_ok())?;
    let postchecked = postcheck(budget);
    account.require(budget, false)?;
    postchecked?;
    account.require(budget, checked.is_ok())?;
    Ok((owner, checked))
}

fn import_and_retain_using<R>(
    request: &Request<'_>,
    input: DecodedCpuInput<'_>,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
    finish: impl FnOnce(
        RetainedProductionConditionalFormulaV2,
        &Request<'_>,
        &mut Budget<'_>,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    retention::retain_reservation_using(budget, RETAINED_STORAGE, |budget| {
        with_decoded_cpu_input(request, input, budget, |request, cpu_input, budget| {
            with_scratch_using(budget, retention::PREPARATION_STORAGE, |budget| {
                let prepared = prepare_v2(request, cpu_input, budget)?;
                require_policy(request, accepted, budget)?;
                budget.charge_work(signature.wire().len())?;
                let retained = import_and_retain_functional_refinement_receipt_v2(
                    prepared.binding,
                    signature,
                    accepted,
                )
                .map_err(|_| Error::Subject("conditional V2 signature/policy"))?;
                let proof = retained.proof();
                let formula = report_for_proof(prepared.binding, prepared.generated_source, proof);
                retention::require_imported_identity(formula, accepted, proof)?;
                current(request.pliron_input(), budget)?;
                let owner = retain(request, formula, cpu_input, retained, accepted.clone());
                finish(owner, request, budget)
            })
        })
    })
}

impl RetainedProductionConditionalFormulaV2 {
    pub const fn report(&self) -> ProductionConditionalFormulaReportV2 {
        self.execution.report
    }
    pub const fn retained_storage_v2(&self) -> usize {
        RETAINED_STORAGE
    }
    /// Policy transport is inert; an independent caller must accept its provenance.
    pub const fn import_policy_v2(&self) -> &FunctionalRefinementImportPolicyV2 {
        &self.policy
    }

    /// Re-encodes and rejoins the same whole input before lending the V2 execution.
    ///
    /// ```compile_fail
    /// use fe2o3_verifier::{RetainedProductionConditionalFormulaV2, ProductionConditionalFormulaExecutionV2};
    /// use fe2o3_verifier::portable_reference_v1::codec::NativeCpuInputV1;
    /// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
    /// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
    /// fn escape<'a>(p: &RetainedProductionConditionalFormulaV2, r: &Request<'_>,
    ///     input: NativeCpuInputV1<'_>, b: &mut Budget<'_>) -> &'a ProductionConditionalFormulaExecutionV2 {
    ///     p.with_replayed_request_v2(r, input, b, |proof, _| Ok(proof)).unwrap()
    /// }
    /// ```
    pub fn with_replayed_request_v2<R>(
        &self,
        request: &Request<'_>,
        input: NativeCpuInputV1<'_>,
        budget: &mut Budget<'_>,
        consume: impl for<'proof> FnOnce(
            &'proof ProductionConditionalFormulaExecutionV2,
            &mut Budget<'_>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.require_reservation(budget)?;
        with_live_cpu(request, input, budget, |request, cpu_input, budget| {
            self.replay_checked(request, cpu_input, budget, consume)
        })
    }

    /// Re-encodes the entire policy-origin input before the same strict replay.
    /// This cannot upgrade a registration-origin receipt or authenticate the
    /// policy description. Original enrollment-owner checks remain external.
    ///
    /// ```compile_fail
    /// use fe2o3_verifier::{RetainedProductionConditionalFormulaV2, ProductionConditionalFormulaExecutionV2};
    /// use fe2o3_verifier::portable_reference_v1::codec::NativeCpuPolicyInputV2;
    /// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
    /// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
    /// fn escape<'a>(p: &RetainedProductionConditionalFormulaV2, r: &Request<'_>,
    ///     input: NativeCpuPolicyInputV2<'_>, b: &mut Budget<'_>) -> &'a ProductionConditionalFormulaExecutionV2 {
    ///     p.with_replayed_policy_request_v2(r, input, b, |proof, _| Ok(proof)).unwrap()
    /// }
    /// ```
    pub fn with_replayed_policy_request_v2<R>(
        &self,
        request: &Request<'_>,
        input: NativeCpuPolicyInputV2<'_>,
        budget: &mut Budget<'_>,
        consume: impl for<'proof> FnOnce(
            &'proof ProductionConditionalFormulaExecutionV2,
            &mut Budget<'_>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.require_reservation(budget)?;
        with_live_policy_cpu(request, input, budget, |request, cpu_input, budget| {
            self.replay_checked(request, cpu_input, budget, consume)
        })
    }

    /// Uses this decoded owner's input and commitment together, without a clone.
    ///
    /// ```compile_fail
    /// use fe2o3_verifier::{RetainedProductionConditionalFormulaV2, ProductionConditionalFormulaExecutionV2};
    /// use fe2o3_verifier::portable_reference_v1::codec::DecodedNativeCpuInputV1;
    /// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
    /// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
    /// fn escape<'a>(p: &RetainedProductionConditionalFormulaV2, r: &Request<'_>,
    ///     input: &DecodedNativeCpuInputV1, b: &mut Budget<'_>) -> &'a ProductionConditionalFormulaExecutionV2 {
    ///     p.with_replayed_decoded_request_v2(r, input, b, |proof, _| Ok(proof)).unwrap()
    /// }
    /// ```
    pub fn with_replayed_decoded_request_v2<R>(
        &self,
        request: &Request<'_>,
        input: &DecodedNativeCpuInputV1,
        budget: &mut Budget<'_>,
        consume: impl for<'proof> FnOnce(
            &'proof ProductionConditionalFormulaExecutionV2,
            &mut Budget<'_>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.require_reservation(budget)?;
        with_decoded_cpu(request, input, budget, |request, cpu_input, budget| {
            self.replay_checked(request, cpu_input, budget, consume)
        })
    }

    /// Borrows the policy input and commitment from one scoped decoded owner.
    /// The original enrollment and source owners must remain current externally.
    ///
    /// ```compile_fail
    /// use fe2o3_verifier::{RetainedProductionConditionalFormulaV2, ProductionConditionalFormulaExecutionV2};
    /// use fe2o3_verifier::portable_reference_v1::codec::DecodedNativeCpuPolicyInputV2;
    /// use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
    /// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
    /// fn escape<'a>(p: &RetainedProductionConditionalFormulaV2, r: &Request<'_>,
    ///     input: &DecodedNativeCpuPolicyInputV2, b: &mut Budget<'_>) -> &'a ProductionConditionalFormulaExecutionV2 {
    ///     p.with_replayed_decoded_policy_request_v2(r, input, b, |proof, _| Ok(proof)).unwrap()
    /// }
    /// ```
    pub fn with_replayed_decoded_policy_request_v2<R>(
        &self,
        request: &Request<'_>,
        input: &DecodedNativeCpuPolicyInputV2,
        budget: &mut Budget<'_>,
        consume: impl for<'proof> FnOnce(
            &'proof ProductionConditionalFormulaExecutionV2,
            &mut Budget<'_>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.require_reservation(budget)?;
        with_decoded_policy_cpu(request, input, budget, |request, cpu_input, budget| {
            self.replay_checked(request, cpu_input, budget, consume)
        })
    }

    fn require_reservation(&self, budget: &mut Budget<'_>) -> Result<(), Error> {
        budget.charge_work(1)?;
        if budget.storage() < RETAINED_STORAGE {
            return Err(Resource::Accounting.into());
        }
        Ok(())
    }

    fn replay_checked<R>(
        &self,
        request: &Request<'_>,
        cpu_input: DigestV1,
        budget: &mut Budget<'_>,
        consume: impl for<'proof> FnOnce(
            &'proof ProductionConditionalFormulaExecutionV2,
            &mut Budget<'_>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        if self.subject != retention::RetainedSubjectV1::from_request(request) {
            return Err(Error::Subject(
                "conditional V2 source/graph/root substitution",
            ));
        }
        require_commitment(self.report().cpu_input, cpu_input)?;
        with_scratch_using(budget, retention::PREPARATION_STORAGE, |budget| {
            let expected = prepare_v2(request, cpu_input, budget)?;
            require_policy(request, &self.policy, budget)?;
            retention::reimport(
                self.execution.report.formula,
                &self.policy,
                self.execution.signed_receipt_wire(),
                &expected,
                budget,
            )?;
            current(request.pliron_input(), budget)?;
            let result = consume(&self.execution, budget);
            current(request.pliron_input(), budget)?;
            result
        })
    }
}

fn retain(
    request: &Request<'_>,
    formula: ProductionConditionalFormulaReportV1,
    cpu_input: DigestV1,
    retained: RetainedImportedFunctionalRefinementReceiptV2,
    policy: FunctionalRefinementImportPolicyV2,
) -> RetainedProductionConditionalFormulaV2 {
    RetainedProductionConditionalFormulaV2 {
        execution: ProductionConditionalFormulaExecutionV2 {
            report: ProductionConditionalFormulaReportV2 { formula, cpu_input },
            retained,
        },
        policy,
        subject: retention::RetainedSubjectV1::from_request(request),
    }
}

fn require_policy(
    request: &Request<'_>,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'_>,
) -> Result<(), Error> {
    budget.charge_work(161)?;
    let [staging] = request
        .pliron_input()
        .retained_policy_checked_refinement_staging()
    else {
        return Err(Error::Subject("conditional V2 staging policy roster"));
    };
    if accepted.boundary() != FunctionalRefinementBoundaryV2::SafeReferenceMirToLivePliron
        || accepted.toolchain() != staging.toolchain()
    {
        return Err(Error::Subject("conditional V2 accepted boundary/toolchain"));
    }
    Ok(())
}

fn reborrow<'a>(input: &NativeCpuInputV1<'a>) -> NativeCpuInputV1<'a> {
    NativeCpuInputV1 {
        association: input.association,
        kernel: input.kernel,
        reference: input.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: input.replay.signature_preimage,
            effect_ir: input.replay.effect_ir,
            effect_ir_sha256: input.replay.effect_ir_sha256,
            observable_output_writes: input.replay.observable_output_writes,
        },
    }
}

fn with_live_cpu<R>(
    request: &Request<'_>,
    input: NativeCpuInputV1<'_>,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    with_encoded_native_cpu_input_v1(reborrow(&input), budget, |_, commitment, budget| {
        with_cpu(
            request,
            CpuCorrespondenceInput::from(input),
            commitment,
            budget,
            run,
        )
    })
    .map_err(Error::Codec)?
}

fn reborrow_policy<'a>(input: &NativeCpuPolicyInputV2<'a>) -> NativeCpuPolicyInputV2<'a> {
    NativeCpuPolicyInputV2 {
        association: input.association,
        kernel: input.kernel,
        reference: input.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: input.replay.signature_preimage,
            effect_ir: input.replay.effect_ir,
            effect_ir_sha256: input.replay.effect_ir_sha256,
            observable_output_writes: input.replay.observable_output_writes,
        },
    }
}

fn with_live_policy_cpu<R>(
    request: &Request<'_>,
    input: NativeCpuPolicyInputV2<'_>,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    with_encoded_native_cpu_policy_input_v2(
        reborrow_policy(&input),
        budget,
        |_, commitment, budget| {
            with_cpu(
                request,
                CpuCorrespondenceInput::from(input),
                commitment,
                budget,
                run,
            )
        },
    )
    .map_err(Error::Codec)?
}

fn with_live_cpu_input<R>(
    request: &Request<'_>,
    input: LiveCpuInput<'_>,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    match input {
        LiveCpuInput::Registration(input) => with_live_cpu(request, input, budget, run),
        LiveCpuInput::Policy(input) => with_live_policy_cpu(request, input, budget, run),
    }
}

enum DecodedCpuInput<'a> {
    Registration(&'a DecodedNativeCpuInputV1),
    Policy(&'a DecodedNativeCpuPolicyInputV2),
}

fn with_decoded_cpu_input<R>(
    request: &Request<'_>,
    input: DecodedCpuInput<'_>,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    match input {
        DecodedCpuInput::Registration(input) => with_decoded_cpu(request, input, budget, run),
        DecodedCpuInput::Policy(input) => with_decoded_policy_cpu(request, input, budget, run),
    }
}

fn with_decoded_policy_cpu<R>(
    request: &Request<'_>,
    input: &DecodedNativeCpuPolicyInputV2,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    with_cpu(
        request,
        CpuCorrespondenceInput::from(input.input_v2()),
        input.commitment_v2(),
        budget,
        run,
    )
}

fn with_decoded_cpu<R>(
    request: &Request<'_>,
    input: &DecodedNativeCpuInputV1,
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    with_cpu(
        request,
        CpuCorrespondenceInput::from(input.input_v1()),
        input.commitment_v1(),
        budget,
        run,
    )
}

// Only the borrowed correspondence subject is shared. Origin stays in each
// complete, domain-separated codec commitment; no synthetic V1 input is made.
struct CpuCorrespondenceInput<'a> {
    semantic_mir_sha256: [u8; 32],
    semantic_root: u32,
    reference: ConditionalReferenceInputV1<'a>,
}

impl<'a> From<NativeCpuInputV1<'a>> for CpuCorrespondenceInput<'a> {
    fn from(input: NativeCpuInputV1<'a>) -> Self {
        Self {
            semantic_mir_sha256: input.association.semantic_mir_sha256,
            semantic_root: input.association.semantic_root,
            reference: ConditionalReferenceInputV1 {
                kernel: input.kernel,
                reference: input.reference,
                replay: input.replay,
            },
        }
    }
}

impl<'a> From<NativeCpuPolicyInputV2<'a>> for CpuCorrespondenceInput<'a> {
    fn from(input: NativeCpuPolicyInputV2<'a>) -> Self {
        Self {
            semantic_mir_sha256: input.association.semantic_mir_sha256,
            semantic_root: input.association.semantic_root,
            reference: ConditionalReferenceInputV1 {
                kernel: input.kernel,
                reference: input.reference,
                replay: input.replay,
            },
        }
    }
}

// This raw commitment is PRIVATE and supplied only by the paired codec paths.
fn with_cpu<R>(
    request: &Request<'_>,
    input: CpuCorrespondenceInput<'_>,
    commitment: [u8; 32],
    budget: &mut Budget<'_>,
    run: impl FnOnce(&Request<'_>, DigestV1, &mut Budget<'_>) -> Result<R, Error>,
) -> Result<R, Error> {
    current(request.pliron_input(), budget)?;
    let semantic = request.source().semantic_ssa().source_semantic();
    budget.charge_work(64)?;
    if input.semantic_mir_sha256 != *semantic.semantic_sha256().as_bytes()
        || input.semantic_mir_sha256
            != *request.pliron_input().source_semantic_identity().as_bytes()
    {
        return Err(Error::Subject("conditional V2 CPU source association"));
    }
    let mut root_found = false;
    for root in semantic.roots() {
        budget.charge_work(1)?;
        root_found |= root.index() == input.semantic_root;
    }
    if !root_found {
        return Err(Error::Subject("conditional V2 CPU root association"));
    }
    with_source_bound_cpu_correspondence_v1(
        request,
        input.reference,
        input.semantic_root,
        budget,
        |cpu, budget| {
            cpu.require_subjects(budget)
                .map_err(Error::Correspondence)?;
            let result = run(
                cpu.request(),
                DigestV1::from_untrusted_bytes(commitment),
                budget,
            );
            current(cpu.request().pliron_input(), budget)?;
            result
        },
    )
    .map_err(Error::Correspondence)?
}

fn require_commitment(expected: DigestV1, actual: DigestV1) -> Result<(), Error> {
    if expected != actual {
        return Err(Error::Subject("conditional V2 CPU input substitution"));
    }
    Ok(())
}

fn prepare_v2(
    request: &Request<'_>,
    cpu_input: DigestV1,
    budget: &mut Budget<'_>,
) -> Result<Prepared, Error> {
    let mut prepared = prepare(request, budget)?;
    budget.charge_work(DOMAIN.len() + 7 * 32)?;
    let obligation = obligation_identity_v2(
        obligation_commitments(request.pliron_input(), prepared.generated_source),
        cpu_input,
    );
    prepared.binding = FunctionalRefinementBindingV2::from_subjects(
        request.pliron_input().reference_subjects(),
        obligation,
    )
    .map_err(|_| Error::Subject("conditional V2 formula binding"))?;
    Ok(prepared)
}

fn obligation_identity_v2(commitments: [DigestV1; 6], cpu_input: DigestV1) -> DigestV1 {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    for identity in commitments.into_iter().chain([cpu_input]) {
        hash.update(identity.as_bytes());
    }
    DigestV1::from_untrusted_bytes(hash.finalize().into())
}

#[cfg(test)]
#[path = "conditional_ranked_formula_cpu_v2_tests.rs"]
mod cpu_tests;
#[cfg(test)]
#[path = "conditional_ranked_formula_fixture_v2_tests.rs"]
mod fixtures;
#[cfg(test)]
#[path = "conditional_ranked_formula_import_check_v2_tests.rs"]
mod import_check_tests;
#[cfg(test)]
#[path = "conditional_ranked_formula_import_v2_tests.rs"]
mod import_tests;
#[cfg(test)]
#[path = "conditional_ranked_formula_policy_v2_tests.rs"]
mod policy_tests;
#[cfg(test)]
#[path = "conditional_ranked_formula_resources_v2_tests.rs"]
mod resource_tests;
