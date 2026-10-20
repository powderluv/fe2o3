//! One-shot application-side audit of the protected compiler current-record endpoint.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use fe2o3_artifact_transaction::InertCompilerExecutionSubjectV1;
#[cfg(target_arch = "x86_64")]
use fe2o3_compiler_closure_capability::ProductionCompilerExecutionDeploymentV1;
use fe2o3_compiler_execution_client::{
    CompilerExecutionClientErrorV1, CompilerExecutionClientV1,
    CompilerExecutionCurrentRecordChallengeV1,
};
use fe2o3_runtime_protocol::{
    COMPILER_EXECUTION_CURRENT_RECORD_ATTESTATION_BYTES_V3,
    COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3,
    CompilerExecutionCurrentRecordAttestationIdentityV3,
    CompilerExecutionCurrentRecordAttestationV3, CompilerExecutionCurrentRecordVerificationErrorV3,
    CompilerExecutionCurrentRecordVerificationIdentityV3,
    CompilerExecutionCurrentRecordVerificationV3, CompilerExecutionIssuerPolicyV1,
    CompilerExecutionReceiptCarriageV1, VerifiedCompilerExecutionCurrentRecordV3,
};

use crate::{
    CompilerGeneratedKernelExpectationRosterV1, CompilerGeneratedKernelExpectationV1,
    WorkerV3AuditorV1, WorkerV3CompilerExecutionEvidenceErrorV1,
    WorkerV3CompilerExecutionVerificationV1, WorkerV3RosterVerificationRequestV1,
    WorkerV3VerificationRequestV1,
};

/// Complete deadline for one application-side current-record verification transaction.
pub const WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1: Duration = Duration::from_secs(30);

/// Lifetime-bound canonical records retained by one admitted FD195 current-record result.
///
/// The private constructor binds both byte arrays and both typed identities to the same already
/// verified attestation. The view cannot outlive or transfer custody from its move-only owner.
/// Copying either byte array creates only inert protocol input: a protected verifier must strictly
/// decode and authenticate the exact pair again under its independently pinned policy, expected
/// fresh challenge, request, and replay policy.
///
/// This view exposes no signing material and grants no service, currentness, verification, load, or
/// launch authority.
///
/// ```compile_fail
/// use fe2o3_host::WorkerV3CompilerCurrentRecordEvidenceViewV1;
/// fn duplicate(view: WorkerV3CompilerCurrentRecordEvidenceViewV1<'_>) {
///     let _second = view.clone();
/// }
/// ```
#[derive(Debug)]
pub struct WorkerV3CompilerCurrentRecordEvidenceViewV1<'evidence> {
    verified: &'evidence VerifiedCompilerExecutionCurrentRecordV3,
}

impl<'evidence> WorkerV3CompilerCurrentRecordEvidenceViewV1<'evidence> {
    const fn from_verified(verified: &'evidence VerifiedCompilerExecutionCurrentRecordV3) -> Self {
        Self { verified }
    }

    /// Returns the domain-separated identity of the exact canonical verification bytes below.
    pub const fn verification_identity(
        &self,
    ) -> CompilerExecutionCurrentRecordVerificationIdentityV3 {
        self.verified.verification().identity()
    }

    /// Returns the domain-separated identity of the exact canonical attestation bytes below.
    pub const fn attestation_identity(
        &self,
    ) -> CompilerExecutionCurrentRecordAttestationIdentityV3 {
        self.verified.attestation().identity()
    }

    /// Returns the exact expected challenge authenticated during admission.
    ///
    /// For a caller-supplied audit this must equal the independently retained caller value. Reading
    /// it from the signed response alone does not establish freshness or replay exclusion.
    pub const fn verification_challenge(&self) -> [u8; 32] {
        self.verified.attestation().challenge()
    }

    /// Borrows the complete canonical current-record verification without transferring custody.
    pub const fn verification_canonical_bytes(
        &self,
    ) -> &[u8; COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3] {
        self.verified.verification().canonical_bytes()
    }

    /// Borrows the complete signed, challenge-bound canonical attestation without transferring
    /// custody.
    pub const fn attestation_canonical_bytes(
        &self,
    ) -> &[u8; COMPILER_EXECUTION_CURRENT_RECORD_ATTESTATION_BYTES_V3] {
        self.verified.attestation().canonical_bytes()
    }

    /// Reports that this borrowed record view grants no final-verifier authority.
    pub const fn grants_verification_authority(&self) -> bool {
        false
    }

    /// Reports that this borrowed record view grants no generic authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }

    /// Reports that this borrowed record view grants no executable-load authority.
    pub const fn grants_load_authority(&self) -> bool {
        false
    }

    /// Reports that this borrowed record view grants no launch authority.
    pub const fn grants_launch_authority(&self) -> bool {
        false
    }
}

/// Move-only signed endpoint evidence for one exact Worker V3 compiler receipt.
///
/// The evidence authenticates a fresh response and external-anchor recovery observation. Ordinary
/// admission uses the receipt's policy, which is not independent production trust. The production
/// factory instead retains fixed-path deployment provenance and verifies against its policy.
/// Neither result grants final verification, load or launch authority or continuous service liveness.
///
/// ```compile_fail
/// use fe2o3_host::WorkerV3CompilerCurrentRecordAuditV1;
/// fn require_clone<T: Clone>() {}
/// require_clone::<WorkerV3CompilerCurrentRecordAuditV1>();
/// ```
#[derive(Debug)]
pub struct WorkerV3CompilerCurrentRecordAuditV1 {
    verified: VerifiedCompilerExecutionCurrentRecordV3,
    #[cfg(target_arch = "x86_64")]
    deployment: Option<ProductionCompilerExecutionDeploymentV1>,
}

impl WorkerV3CompilerCurrentRecordAuditV1 {
    fn ordinary(verified: VerifiedCompilerExecutionCurrentRecordV3) -> Self {
        Self {
            verified,
            #[cfg(target_arch = "x86_64")]
            deployment: None,
        }
    }

    /// Rechecks retained installed production configuration, not current service liveness.
    ///
    /// Ordinary audits and canonical-byte imports cannot acquire this provenance. This does not
    /// perform a second fresh service exchange or grant verification, load or launch authority.
    #[cfg(target_arch = "x86_64")]
    pub fn revalidate_production_deployment(
        &self,
    ) -> Result<(), WorkerV3CompilerCurrentRecordAuditErrorV1> {
        self.deployment
            .as_ref()
            .ok_or(WorkerV3CompilerCurrentRecordAuditErrorV1::MissingProductionDeployment)?
            .revalidate()
            .map_err(WorkerV3CompilerCurrentRecordAuditErrorV1::ProductionDeployment)
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn revalidate_optional_production_deployment(&self) -> Result<(), String> {
        if let Some(deployment) = &self.deployment {
            deployment.revalidate()?;
        }
        Ok(())
    }

    /// Borrows the exact canonical records authenticated by this move-only audit.
    ///
    /// The returned view is authority-free and cannot outlive this owner. A downstream protected
    /// verifier must independently authenticate the byte pair rather than treating the host's
    /// identities or currentness booleans as authority.
    pub const fn canonical_evidence_view(&self) -> WorkerV3CompilerCurrentRecordEvidenceViewV1<'_> {
        WorkerV3CompilerCurrentRecordEvidenceViewV1::from_verified(&self.verified)
    }

    pub const fn verification(&self) -> &CompilerExecutionCurrentRecordVerificationV3 {
        self.verified.verification()
    }

    pub const fn attestation_identity(
        &self,
    ) -> CompilerExecutionCurrentRecordAttestationIdentityV3 {
        self.verified.attestation().identity()
    }

    pub const fn authenticates_pinned_signing_key(&self) -> bool {
        self.verified.authenticates_pinned_signing_key()
    }

    pub const fn authenticates_expected_fresh_challenge(&self) -> bool {
        self.verified.authenticates_expected_challenge()
    }

    pub const fn authenticates_protected_key_custody(&self) -> bool {
        false
    }

    pub const fn authenticates_protected_current_record(&self) -> bool {
        self.verified.authenticates_protected_current_record()
    }

    pub const fn authenticates_external_anchor_commit(&self) -> bool {
        self.verified.authenticates_external_anchor_commit()
    }

    pub const fn external_rollback_verification_identity(&self) -> [u8; 32] {
        self.verified.external_rollback_verification_identity()
    }

    pub const fn authenticates_external_rollback_currentness(&self) -> bool {
        self.verified.authenticates_external_rollback_currentness()
    }

    pub const fn grants_verification_authority(&self) -> bool {
        false
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }

    pub const fn grants_load_authority(&self) -> bool {
        false
    }

    pub const fn grants_launch_authority(&self) -> bool {
        false
    }

    /// Consumes this one-use audit into an exact compiler-execution evidence lane.
    ///
    /// Every V3 current-record coordinate is compared with the supplied subject and complete
    /// carriage before the signed attestation is retained. This transition still grants no final
    /// verification, load, or launch authority. Retained production deployment, when present, is
    /// revalidated before binding; the remaining refinement and invocation obligations stay separate.
    pub fn bind_exact_compiler_execution_v1(
        self,
        subject: &InertCompilerExecutionSubjectV1,
        carriage: &CompilerExecutionReceiptCarriageV1,
    ) -> Result<WorkerV3CompilerExecutionVerificationV1, WorkerV3CompilerExecutionEvidenceErrorV1>
    {
        WorkerV3CompilerExecutionVerificationV1::from_current_record_audit(subject, carriage, self)
    }
}

/// Admits exact canonical compiler current-record evidence inside a future protected Worker V3
/// backend.
///
/// The caller supplies an independently pinned policy and a cryptographically fresh challenge
/// whose replay exclusion it owns. Both canonical records are decoded independently, the nested
/// verification must byte-match the separately transported verification, and the signed
/// attestation is checked against the exact subject, carriage, policy, challenge, Worker-ledger
/// position, and external rollback-currentness observation. The host repeats the exact coordinate
/// comparisons before constructing the move-only compiler-execution lane.
///
/// This crate-internal bridge does not authenticate the transport, establish protected key
/// custody, or grant verification, load, or launch authority. It must remain unreachable to
/// applications until a production backend supplies a move-only authenticated measured-service
/// session and owns those unsafe obligations.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn admit_worker_v3_compiler_current_record_evidence_v1(
    pinned_policy: &CompilerExecutionIssuerPolicyV1,
    subject: &InertCompilerExecutionSubjectV1,
    carriage: &CompilerExecutionReceiptCarriageV1,
    expected_fresh_challenge: [u8; 32],
    canonical_verification: &[u8],
    canonical_attestation: &[u8],
) -> Result<
    WorkerV3CompilerExecutionVerificationV1,
    WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1,
> {
    if carriage.request().subject() != subject {
        return Err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::RequestMismatch);
    }
    if carriage.policy() != pinned_policy {
        return Err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::PolicyMismatch);
    }

    let verification = CompilerExecutionCurrentRecordVerificationV3::decode(canonical_verification)
        .map_err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Verification)?;
    let attestation = CompilerExecutionCurrentRecordAttestationV3::decode(canonical_attestation)
        .map_err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation)?;
    if attestation.verification() != &verification
        || attestation.verification().canonical_bytes().as_slice() != canonical_verification
    {
        return Err(
            WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::VerificationAttestationMismatch,
        );
    }

    let verified = attestation
        .verify(pinned_policy, carriage, expected_fresh_challenge)
        .map_err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation)?;
    independently_recheck_current_record_v1(
        pinned_policy,
        subject,
        carriage,
        expected_fresh_challenge,
        &verified,
    )?;

    WorkerV3CompilerExecutionVerificationV1::from_current_record_audit(
        subject,
        carriage,
        WorkerV3CompilerCurrentRecordAuditV1::ordinary(verified),
    )
    .map_err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Evidence)
}

fn independently_recheck_current_record_v1(
    pinned_policy: &CompilerExecutionIssuerPolicyV1,
    subject: &InertCompilerExecutionSubjectV1,
    carriage: &CompilerExecutionReceiptCarriageV1,
    expected_fresh_challenge: [u8; 32],
    verified: &VerifiedCompilerExecutionCurrentRecordV3,
) -> Result<(), WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1> {
    let attestation = verified.attestation();
    let verification = verified.verification();
    for (matches, field) in [
        (
            attestation.challenge() == expected_fresh_challenge,
            "fresh compiler current-record challenge",
        ),
        (
            attestation.verifying_key() == *pinned_policy.verifying_key(),
            "pinned compiler current-record signing key",
        ),
        (
            verification.policy_identity() == *pinned_policy.identity().as_bytes(),
            "compiler-execution policy",
        ),
        (
            verification.subject_identity() == *subject.identity().sha256(),
            "compiler-execution subject",
        ),
        (
            verification.carriage_identity() == *carriage.identity().as_bytes(),
            "compiler-execution carriage",
        ),
        (
            verification.issuer_journal_identity()
                == carriage.acknowledgment().issuer_journal_identity(),
            "compiler-execution issuer journal",
        ),
        (
            verification.worker_ledger_record_identity()
                == carriage.acknowledgment().worker_ledger_record_identity(),
            "compiler-execution Worker ledger record",
        ),
        (
            verification.sequence() == carriage.acknowledgment().sequence(),
            "compiler-execution rollback sequence",
        ),
        (
            verification.prior_rollback_anchor()
                == carriage.publication().receipt().prior_rollback_anchor(),
            "compiler-execution prior rollback anchor",
        ),
        (
            verification.current_rollback_anchor()
                == carriage.acknowledgment().current_rollback_anchor(),
            "compiler-execution current rollback anchor",
        ),
        (
            verification.external_anchor_verifying_key()
                == *pinned_policy.external_anchor_verifying_key(),
            "pinned external rollback signing key",
        ),
    ] {
        if !matches {
            return Err(
                WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::IdentityMismatch(field),
            );
        }
    }

    let expected_currentness =
        CompilerExecutionCurrentRecordVerificationV3::external_anchor_currentness_challenge(
            carriage,
            verification.external_anchor_commit_receipt(),
            expected_fresh_challenge,
        )
        .map_err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Verification)?;
    if verification
        .external_anchor_currentness_receipt()
        .challenge()
        != &expected_currentness
    {
        return Err(
            WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::StaleExternalRollbackCurrentness,
        );
    }
    Ok(())
}

/// One-use auditor owning an inherited endpoint or an explicitly owned application peer.
///
/// ```compile_fail
/// use fe2o3_host::InheritedWorkerV3CompilerCurrentRecordAuditorV1;
/// fn require_clone<T: Clone>() {}
/// require_clone::<InheritedWorkerV3CompilerCurrentRecordAuditorV1>();
/// ```
pub struct InheritedWorkerV3CompilerCurrentRecordAuditorV1 {
    session: Option<CompilerCurrentRecordSessionV1>,
}

struct CompilerCurrentRecordSessionV1 {
    client: CompilerExecutionClientV1,
    #[cfg(target_arch = "x86_64")]
    deployment: Option<ProductionCompilerExecutionDeploymentV1>,
}

impl fmt::Debug for InheritedWorkerV3CompilerCurrentRecordAuditorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InheritedWorkerV3CompilerCurrentRecordAuditorV1")
            .field("available", &self.session.is_some())
            .field("authority", &"none")
            .finish()
    }
}

impl InheritedWorkerV3CompilerCurrentRecordAuditorV1 {
    /// Consumes an owned application peer into a private close-on-exec audit session.
    ///
    /// A fresh close-on-exec duplicate is retained before the original descriptor is dropped.
    /// The input is consumed on success, error, or unwind; no fixed descriptor slot is inspected
    /// or reacquired. Admission uses the same connected unnamed Unix `SOCK_SEQPACKET` checks and
    /// fixed deadline as the inherited entry point.
    ///
    /// Ownership does not establish inherited or production provenance, exclusive ownership of
    /// the underlying kernel object, protected-key custody, or verification/load/launch authority.
    /// Existing aliases may interfere with transport; signed request, policy, and challenge checks
    /// remain mandatory. Production deployment admission remains a separate boundary.
    ///
    /// Callers cannot supply a different deadline:
    ///
    /// ```compile_fail,E0061
    /// use fe2o3_host::InheritedWorkerV3CompilerCurrentRecordAuditorV1;
    /// fn arbitrary_deadline(peer: std::os::fd::OwnedFd) {
    ///     let _ = InheritedWorkerV3CompilerCurrentRecordAuditorV1::admit_owned_application_service(
    ///         peer, std::time::Duration::from_secs(3600),
    ///     );
    /// }
    /// ```
    pub fn admit_owned_application_service(
        peer: std::os::fd::OwnedFd,
    ) -> Result<Self, CompilerExecutionClientErrorV1> {
        let retained = rustix::io::fcntl_dupfd_cloexec(&peer, 3)
            .map_err(|error| CompilerExecutionClientErrorV1::Descriptor(error.into()))?;
        drop(peer);
        CompilerExecutionClientV1::admit(
            retained,
            WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1,
        )
        .map(Self::from_client)
    }

    /// Consumes the inherited public FD slot into one private close-on-exec client.
    ///
    /// # Safety
    /// Transfer exclusive ownership of inherited FD 195, with no existing Rust
    /// owner or outstanding borrow and no thread, handler, or foreign code able to
    /// close, replace, or acquire it during admission. An absent slot must remain
    /// unallocated until return. This transfer is consumed once, including failure
    /// or unwind; descriptor checks do not prove that it is an inherited input.
    ///
    /// ```compile_fail,E0133
    /// use fe2o3_host::InheritedWorkerV3CompilerCurrentRecordAuditorV1;
    /// let _ = InheritedWorkerV3CompilerCurrentRecordAuditorV1::admit_inherited_application_service();
    /// ```
    pub unsafe fn admit_inherited_application_service()
    -> Result<Self, CompilerExecutionClientErrorV1> {
        // SAFETY: the caller transfers exactly the custody required by the client.
        unsafe {
            CompilerExecutionClientV1::admit_inherited_child(
                WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1,
            )
        }
        .map(Self::from_client)
    }

    fn from_client(client: CompilerExecutionClientV1) -> Self {
        Self {
            session: Some(CompilerCurrentRecordSessionV1 {
                client,
                #[cfg(target_arch = "x86_64")]
                deployment: None,
            }),
        }
    }

    /// Consumes FD195 and independently admits the fixed root-owned production deployment.
    ///
    /// No caller policy, path or sealed capability can supply production provenance. All failures
    /// after FD admission close that endpoint; successful audit moves its original configuration
    /// owners alongside the challenge-bound response. This is not a service-liveness lease.
    ///
    /// # Safety
    /// Transfers exclusive ownership of inherited FD 195 under the same descriptor
    /// custody requirements as [`Self::admit_inherited_application_service`].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn admit_production_application_service()
    -> Result<Self, WorkerV3CompilerCurrentRecordAuditErrorV1> {
        // SAFETY: forwards the caller's exclusive inherited-descriptor transfer.
        unsafe {
            Self::admit_production_application_service_with_timeout(
                WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1,
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) unsafe fn admit_production_application_service_with_timeout(
        timeout: std::time::Duration,
    ) -> Result<Self, WorkerV3CompilerCurrentRecordAuditErrorV1> {
        // SAFETY: the caller transfers the inherited slot exactly once.
        let client = unsafe {
            CompilerExecutionClientV1::admit_inherited_child(
                timeout.min(WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1),
            )
        }
        .map_err(WorkerV3CompilerCurrentRecordAuditErrorV1::Client)?;
        let deployment = ProductionCompilerExecutionDeploymentV1::open()
            .map_err(WorkerV3CompilerCurrentRecordAuditErrorV1::ProductionDeployment)?;
        Ok(Self {
            session: Some(CompilerCurrentRecordSessionV1 {
                client,
                deployment: Some(deployment),
            }),
        })
    }

    /// Audits the exact compiler current record retained by one aggregate roster request.
    ///
    /// The inherited endpoint remains one-use and is consumed on every success or failure. The
    /// returned move-only audit authenticates the signed current-record response but grants no
    /// protected-key, verification, load, or launch authority.
    pub fn audit_roster<R>(
        &mut self,
        request: &WorkerV3RosterVerificationRequestV1<'_, R>,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    where
        R: CompilerGeneratedKernelExpectationRosterV1,
    {
        self.audit_exact(
            request.compiler_execution_subject(),
            request.compiler_execution_receipt_carriage(),
        )
    }

    /// Audits the exact V53 publication under its retained current artifact token.
    /// The result authenticates the compiler current record only, not the
    /// separate native proof, protected-key custody, load, or launch gates.
    pub fn audit_mixed_v53<R>(
        &mut self,
        request: &crate::MixedWorkerV53VerificationRequest<'_, R>,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    where
        R: CompilerGeneratedKernelExpectationRosterV1,
    {
        if let Err(error) = request.revalidate_currentness() {
            self.session.take();
            return Err(WorkerV3CompilerCurrentRecordAuditErrorV1::MixedCurrentness(
                error,
            ));
        }
        let result = self.audit_exact(
            request.admission().compiler_execution_subject(),
            request.admission().compiler_execution_receipt(),
        );
        request
            .revalidate_currentness()
            .map_err(WorkerV3CompilerCurrentRecordAuditErrorV1::MixedCurrentness)?;
        result
    }

    /// Audits the exact V89 publication under its retained current artifact token.
    /// This authenticates the current record only, not proof or execution authority.
    pub fn audit_mixed_v89<R>(
        &mut self,
        request: &crate::MixedWorkerV89VerificationRequest<'_, R>,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    where
        R: CompilerGeneratedKernelExpectationRosterV1,
    {
        if let Err(error) = request.revalidate_currentness() {
            self.session.take();
            return Err(WorkerV3CompilerCurrentRecordAuditErrorV1::MixedCurrentness(
                error,
            ));
        }
        let result = self.audit_exact(
            request.admission().compiler_execution_subject(),
            request.admission().compiler_execution_receipt(),
        );
        request
            .revalidate_currentness()
            .map_err(WorkerV3CompilerCurrentRecordAuditErrorV1::MixedCurrentness)?;
        result
    }

    /// Audits one aggregate request using a caller-owned expected challenge.
    ///
    /// The challenge is consumed with the one-use FD195 endpoint. Its caller remains responsible
    /// for cryptographic freshness, uniqueness, and replay exclusion. The returned evidence and its
    /// canonical byte view remain authority-free.
    pub fn audit_roster_with_challenge<R>(
        &mut self,
        request: &WorkerV3RosterVerificationRequestV1<'_, R>,
        expected_challenge: CompilerExecutionCurrentRecordChallengeV1,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    where
        R: CompilerGeneratedKernelExpectationRosterV1,
    {
        self.audit_exact_with_challenge(
            request.compiler_execution_subject(),
            request.compiler_execution_receipt_carriage(),
            expected_challenge,
        )
    }

    /// Audits one singleton request using a caller-owned expected challenge.
    ///
    /// This is the singleton counterpart of [`Self::audit_roster_with_challenge`].
    pub fn audit_with_challenge<K>(
        &mut self,
        request: &WorkerV3VerificationRequestV1<'_, K>,
        expected_challenge: CompilerExecutionCurrentRecordChallengeV1,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    where
        K: CompilerGeneratedKernelExpectationV1,
    {
        self.audit_exact_with_challenge(
            request.compiler_execution_subject(),
            request.compiler_execution_receipt_carriage(),
            expected_challenge,
        )
    }

    fn audit_exact(
        &mut self,
        subject: &InertCompilerExecutionSubjectV1,
        carriage: &CompilerExecutionReceiptCarriageV1,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    {
        self.audit_exact_inner(subject, carriage, None)
    }

    fn audit_exact_with_challenge(
        &mut self,
        subject: &InertCompilerExecutionSubjectV1,
        carriage: &CompilerExecutionReceiptCarriageV1,
        expected_challenge: CompilerExecutionCurrentRecordChallengeV1,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    {
        self.audit_exact_inner(subject, carriage, Some(expected_challenge))
    }

    fn audit_exact_inner(
        &mut self,
        subject: &InertCompilerExecutionSubjectV1,
        carriage: &CompilerExecutionReceiptCarriageV1,
        challenge: Option<CompilerExecutionCurrentRecordChallengeV1>,
    ) -> Result<WorkerV3CompilerCurrentRecordAuditV1, WorkerV3CompilerCurrentRecordAuditErrorV1>
    {
        use WorkerV3CompilerCurrentRecordAuditErrorV1 as E;
        let session = self.session.take().ok_or(E::AlreadyConsumed)?;
        if carriage.request().subject() != subject {
            return Err(E::RequestMismatch);
        }
        let policy = carriage.policy();
        #[cfg(target_arch = "x86_64")]
        let policy = if let Some(deployment) = &session.deployment {
            deployment.revalidate().map_err(E::ProductionDeployment)?;
            if deployment.policy() != policy {
                return Err(E::PolicyMismatch);
            }
            deployment.policy()
        } else {
            policy
        };
        let transaction = match challenge {
            Some(challenge) => session.client.verify_current_only_with_challenge(
                policy,
                carriage.clone(),
                challenge,
            ),
            None => session.client.verify_current_only(policy, carriage.clone()),
        };
        // Revalidate even on a failed exchange, before reporting its transport/signature error.
        #[cfg(target_arch = "x86_64")]
        if let Some(deployment) = &session.deployment {
            deployment.revalidate().map_err(E::ProductionDeployment)?;
        }
        Ok(WorkerV3CompilerCurrentRecordAuditV1 {
            verified: transaction.map_err(E::Client)?,
            #[cfg(target_arch = "x86_64")]
            deployment: session.deployment,
        })
    }
}

impl<K: CompilerGeneratedKernelExpectationV1> WorkerV3AuditorV1<K>
    for InheritedWorkerV3CompilerCurrentRecordAuditorV1
{
    type Error = WorkerV3CompilerCurrentRecordAuditErrorV1;
    type Evidence = WorkerV3CompilerCurrentRecordAuditV1;

    fn audit(
        &mut self,
        request: &WorkerV3VerificationRequestV1<'_, K>,
    ) -> Result<Self::Evidence, Self::Error> {
        self.audit_exact(
            request.compiler_execution_subject(),
            request.compiler_execution_receipt_carriage(),
        )
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3CompilerCurrentRecordAuditErrorV1 {
    AlreadyConsumed,
    RequestMismatch,
    PolicyMismatch,
    MissingProductionDeployment,
    ProductionDeployment(String),
    Client(CompilerExecutionClientErrorV1),
    MixedCurrentness(crate::RecoveredWorkerV3AdmissionErrorV1),
}

/// Failure to admit canonical current-record evidence into the Worker V3 host lane.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1 {
    RequestMismatch,
    PolicyMismatch,
    Verification(CompilerExecutionCurrentRecordVerificationErrorV3),
    Attestation(CompilerExecutionCurrentRecordVerificationErrorV3),
    VerificationAttestationMismatch,
    IdentityMismatch(&'static str),
    StaleExternalRollbackCurrentness,
    Evidence(WorkerV3CompilerExecutionEvidenceErrorV1),
}

impl fmt::Display for WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestMismatch => formatter.write_str(
                "compiler current-record evidence subject differs from its receipt carriage",
            ),
            Self::PolicyMismatch => formatter.write_str(
                "compiler current-record evidence carriage differs from the pinned policy",
            ),
            Self::Verification(error) => {
                write!(formatter, "compiler current-record verification failed: {error}")
            }
            Self::Attestation(error) => {
                write!(formatter, "compiler current-record attestation failed: {error}")
            }
            Self::VerificationAttestationMismatch => formatter.write_str(
                "compiler current-record attestation embeds a different verification",
            ),
            Self::IdentityMismatch(field) => {
                write!(formatter, "compiler current-record {field} identity mismatch")
            }
            Self::StaleExternalRollbackCurrentness => formatter.write_str(
                "compiler current-record external rollback observation is stale for the expected challenge",
            ),
            Self::Evidence(error) => write!(
                formatter,
                "compiler current-record evidence could not enter the Worker V3 lane: {error}"
            ),
        }
    }
}

impl Error for WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Verification(error) | Self::Attestation(error) => Some(error),
            Self::Evidence(error) => Some(error),
            Self::RequestMismatch
            | Self::PolicyMismatch
            | Self::VerificationAttestationMismatch
            | Self::IdentityMismatch(_)
            | Self::StaleExternalRollbackCurrentness => None,
        }
    }
}

impl fmt::Display for WorkerV3CompilerCurrentRecordAuditErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyConsumed => {
                formatter.write_str("compiler current-record auditor was already consumed")
            }
            Self::RequestMismatch => formatter.write_str(
                "compiler current-record audit subject differs from its receipt carriage",
            ),
            Self::Client(error) => {
                write!(formatter, "compiler current-record service failed: {error}")
            }
            Self::MixedCurrentness(error) => {
                write!(
                    formatter,
                    "mixed Worker current publication changed: {error}"
                )
            }
            Self::PolicyMismatch => formatter
                .write_str("compiler receipt policy differs from the installed production policy"),
            Self::MissingProductionDeployment => formatter
                .write_str("compiler audit has no installed production deployment provenance"),
            Self::ProductionDeployment(error) => write!(
                formatter,
                "compiler production deployment changed or could not be admitted: {error}"
            ),
        }
    }
}

impl Error for WorkerV3CompilerCurrentRecordAuditErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Client(error) => Some(error),
            Self::MixedCurrentness(error) => Some(error),
            Self::AlreadyConsumed
            | Self::RequestMismatch
            | Self::PolicyMismatch
            | Self::MissingProductionDeployment
            | Self::ProductionDeployment(_) => None,
        }
    }
}

include!("compiler_execution_current_record_audit/tests.rs");
