#[cfg(test)]
mod tests {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::thread;

    use ed25519_dalek::{Signer, SigningKey};
    use fe2o3_artifact_transaction::{
        INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V1, INERT_COMPILER_EXECUTION_SUBJECT_MAGIC_V1,
        INERT_COMPILER_EXECUTION_SUBJECT_VERSION_V1,
    };
    use fe2o3_external_anchor_protocol::{
        AnchorPositionV1, AnchorTransitionReceiptV1, AnchoredStateV1, CallerNonceV1,
        HashChainHeadV1, PinnedAnchorKeyV1, UnsignedAnchorObservationV1,
    };
    use fe2o3_runtime_protocol::{
        COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3,
        CompilerExecutionAttestationChallengeV1, CompilerExecutionAttestationReceiptV1,
        CompilerExecutionAttestationRequestV1, CompilerExecutionCurrentRecordAttestationV3,
        CompilerExecutionCurrentRecordVerificationV3, CompilerExecutionExternalAnchorTransactionV1,
        CompilerExecutionIssuerMeasurementV1, CompilerExecutionIssuerPolicyV1,
        CompilerExecutionReceiptPublicationAckV1, CompilerExecutionReceiptPublicationV1,
        CompilerExecutionServiceRequestKindV1, CompilerExecutionServiceRequestV1,
        CompilerExecutionServiceResponseV1, MAX_COMPILER_EXECUTION_SERVICE_REQUEST_BYTES_V1,
    };
    use sha2::{Digest, Sha256};

    use super::*;

    const SUBJECT_IDENTITY_DOMAIN: &[u8] = b"FE2O3/INERT-COMPILER-EXECUTION-SUBJECT/V1\0";
    const COMPILER_CLOSURE_IDENTITY_DOMAIN: &[u8] = b"fe2o3-compiler-closure-identity-v2\0";
    const CURRENT_RECORD_VERIFICATION_IDENTITY_DOMAIN: &[u8] =
        b"FE2O3/COMPILER-EXECUTION-CURRENT-RECORD-VERIFICATION/V3\0";
    const CURRENT_RECORD_ATTESTATION_IDENTITY_DOMAIN: &[u8] =
        b"FE2O3/COMPILER-EXECUTION-CURRENT-RECORD-ATTESTATION/V3\0";
    const CURRENT_RECORD_ATTESTATION_SIGNATURE_DOMAIN: &[u8] =
        b"FE2O3/COMPILER-EXECUTION-CURRENT-RECORD-ATTESTATION-SIGNATURE/V3\0";
    const CURRENT_RECORD_HEADER_BYTES: usize = 24;
    const CURRENT_RECORD_ATTESTATION_VERIFICATION_OFFSET: usize = CURRENT_RECORD_HEADER_BYTES + 32;
    const CURRENT_RECORD_ATTESTATION_SIGNED_PREFIX_BYTES: usize =
        CURRENT_RECORD_ATTESTATION_VERIFICATION_OFFSET
            + COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3
            + 32;
    const CURRENT_RECORD_ATTESTATION_PREIMAGE_BYTES: usize =
        CURRENT_RECORD_ATTESTATION_SIGNED_PREFIX_BYTES + 64;
    const CURRENT_RECORD_WORKER_LEDGER_OFFSET: usize = CURRENT_RECORD_HEADER_BYTES + 4 * 32;
    const CURRENT_RECORD_CURRENT_ROLLBACK_ANCHOR_OFFSET: usize =
        CURRENT_RECORD_HEADER_BYTES + 5 * 32 + 8 + 32;

    struct Fixture {
        signing_key: SigningKey,
        anchor_signing_key: SigningKey,
        policy: CompilerExecutionIssuerPolicyV1,
        subject: InertCompilerExecutionSubjectV1,
        carriage: CompilerExecutionReceiptCarriageV1,
    }

    impl Fixture {
        fn new(subject_seed: u8) -> Self {
            Self::with_signing_seed(subject_seed, 0x51)
        }

        fn with_signing_seed(subject_seed: u8, signing_seed: u8) -> Self {
            let signing_key = SigningKey::from_bytes(&[signing_seed; 32]);
            let anchor_signing_key = SigningKey::from_bytes(&[signing_seed + 1; 32]);
            let policy = CompilerExecutionIssuerPolicyV1::new(
                7,
                CompilerExecutionIssuerMeasurementV1::new([0x61; 32], 123).unwrap(),
                CompilerExecutionIssuerMeasurementV1::new([0x62; 32], 456).unwrap(),
                signing_key.verifying_key().to_bytes(),
                anchor_signing_key.verifying_key().to_bytes(),
            )
            .unwrap();
            let subject = subject(subject_seed);
            let challenge = CompilerExecutionAttestationChallengeV1::new(
                &policy, &subject, [0x63; 32], 1, [0; 32],
            )
            .unwrap();
            let request =
                CompilerExecutionAttestationRequestV1::new(challenge, subject.clone()).unwrap();
            let receipt =
                CompilerExecutionAttestationReceiptV1::issue(&policy, &request, &signing_key)
                    .unwrap();
            let publication =
                CompilerExecutionReceiptPublicationV1::new([0x64; 32], [0x65; 32], receipt)
                    .unwrap();
            let acknowledgment =
                CompilerExecutionReceiptPublicationAckV1::new(&publication, [0x66; 32]).unwrap();
            let carriage = CompilerExecutionReceiptCarriageV1::new(
                policy.clone(),
                request,
                publication,
                acknowledgment,
            )
            .unwrap();
            Self {
                signing_key,
                anchor_signing_key,
                policy,
                subject,
                carriage,
            }
        }

        fn anchor_receipt(&self) -> AnchorTransitionReceiptV1 {
            let transaction = CompilerExecutionExternalAnchorTransactionV1::new(
                self.policy.clone(),
                self.carriage.request().clone(),
                self.carriage.publication().clone(),
            )
            .unwrap();
            let key =
                PinnedAnchorKeyV1::from_bytes(self.anchor_signing_key.verifying_key().to_bytes())
                    .unwrap();
            let pending =
                AnchoredStateV1::from_local_state(0, HashChainHeadV1::from_bytes([0; 32]))
                    .prepare(transaction.external_anchor_digest(), &key)
                    .unwrap()
                    .begin_advance(CallerNonceV1::from_bytes([0x67; 32]), &key)
                    .unwrap();
            let unsigned = UnsignedAnchorObservationV1::from_challenge(
                pending.challenge(),
                AnchorPositionV1::Proposed,
            );
            let signature = self.anchor_signing_key.sign(&unsigned.signing_bytes());
            AnchorTransitionReceiptV1::new(
                pending.challenge().clone(),
                &unsigned.attach_signature(signature.to_bytes()),
                &key,
            )
            .unwrap()
        }
    }

    fn direct_current_record_attestation(
        fixture: &Fixture,
        verification_challenge: [u8; 32],
    ) -> CompilerExecutionCurrentRecordAttestationV3 {
        let commit_receipt = fixture.anchor_receipt();
        let currentness_challenge =
            CompilerExecutionCurrentRecordVerificationV3::external_anchor_currentness_challenge(
                &fixture.carriage,
                &commit_receipt,
                verification_challenge,
            )
            .unwrap();
        let anchor_key =
            PinnedAnchorKeyV1::from_bytes(fixture.anchor_signing_key.verifying_key().to_bytes())
                .unwrap();
        let unsigned = UnsignedAnchorObservationV1::from_challenge(
            &currentness_challenge,
            AnchorPositionV1::Proposed,
        );
        let signature = fixture.anchor_signing_key.sign(&unsigned.signing_bytes());
        let currentness_receipt = AnchorTransitionReceiptV1::new(
            currentness_challenge,
            &unsigned.attach_signature(signature.to_bytes()),
            &anchor_key,
        )
        .unwrap();
        let verification = CompilerExecutionCurrentRecordVerificationV3::new(
            &fixture.carriage,
            commit_receipt,
            currentness_receipt,
            verification_challenge,
            [0x91; 32],
            [0x92; 32],
        )
        .unwrap();
        CompilerExecutionCurrentRecordAttestationV3::issue(
            &fixture.policy,
            &fixture.carriage,
            verification,
            verification_challenge,
            &fixture.signing_key,
        )
        .unwrap()
    }

    fn direct_current_record_audit(
        fixture: &Fixture,
        verification_challenge: [u8; 32],
    ) -> WorkerV3CompilerCurrentRecordAuditV1 {
        let attestation = direct_current_record_attestation(fixture, verification_challenge);
        let verified = attestation
            .verify(&fixture.policy, &fixture.carriage, verification_challenge)
            .unwrap();
        WorkerV3CompilerCurrentRecordAuditV1::ordinary(verified)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ordinary_and_canonical_audits_have_no_production_provenance() {
        let fixture = Fixture::new(0x20);
        let audit = direct_current_record_audit(&fixture, [0x91; 32]);
        assert!(matches!(
            audit.revalidate_production_deployment(),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::MissingProductionDeployment)
        ));
        let bound = audit
            .bind_exact_compiler_execution_v1(&fixture.subject, &fixture.carriage)
            .unwrap();
        assert!(matches!(
            bound.revalidate_production_deployment(),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::MissingProductionDeployment)
        ));
        let (verification, attestation) = canonical_current_record_evidence(&fixture, [0x92; 32]);
        let imported = admit_worker_v3_compiler_current_record_evidence_v1(
            &fixture.policy,
            &fixture.subject,
            &fixture.carriage,
            [0x92; 32],
            &verification,
            &attestation,
        )
        .unwrap();
        assert!(matches!(
            imported.revalidate_production_deployment(),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::MissingProductionDeployment)
        ));
    }

    #[cfg(target_arch = "x86_64")]
    #[path = "production.rs"]
    mod production;

    #[path = "owned_input.rs"]
    mod owned_input;

    fn canonical_current_record_evidence(
        fixture: &Fixture,
        verification_challenge: [u8; 32],
    ) -> (Vec<u8>, Vec<u8>) {
        let attestation = direct_current_record_attestation(fixture, verification_challenge);
        (
            attestation.verification().canonical_bytes().to_vec(),
            attestation.canonical_bytes().to_vec(),
        )
    }

    fn domain_identity(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
        digest.finalize().into()
    }

    fn rebuild_current_record_attestation(
        fixture: &Fixture,
        mut attestation: Vec<u8>,
        challenge: [u8; 32],
        verification: &[u8],
    ) -> Vec<u8> {
        assert_eq!(
            verification.len(),
            COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3
        );
        attestation[CURRENT_RECORD_HEADER_BYTES..CURRENT_RECORD_ATTESTATION_VERIFICATION_OFFSET]
            .copy_from_slice(&challenge);
        attestation[CURRENT_RECORD_ATTESTATION_VERIFICATION_OFFSET
            ..CURRENT_RECORD_ATTESTATION_VERIFICATION_OFFSET
                + COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3]
            .copy_from_slice(verification);
        let signature_message = domain_identity(
            CURRENT_RECORD_ATTESTATION_SIGNATURE_DOMAIN,
            &attestation[..CURRENT_RECORD_ATTESTATION_SIGNED_PREFIX_BYTES],
        );
        let signature = fixture.signing_key.sign(&signature_message).to_bytes();
        attestation[CURRENT_RECORD_ATTESTATION_SIGNED_PREFIX_BYTES
            ..CURRENT_RECORD_ATTESTATION_PREIMAGE_BYTES]
            .copy_from_slice(&signature);
        let identity = domain_identity(
            CURRENT_RECORD_ATTESTATION_IDENTITY_DOMAIN,
            &attestation[..CURRENT_RECORD_ATTESTATION_PREIMAGE_BYTES],
        );
        attestation[CURRENT_RECORD_ATTESTATION_PREIMAGE_BYTES..].copy_from_slice(&identity);
        attestation
    }

    fn mutate_verification_identity_field(
        fixture: &Fixture,
        challenge: [u8; 32],
        verification: &[u8],
        attestation: &[u8],
        offset: usize,
        replacement: [u8; 32],
    ) -> (Vec<u8>, Vec<u8>) {
        let mut verification = verification.to_vec();
        verification[offset..offset + 32].copy_from_slice(&replacement);
        let identity_offset = verification.len() - 32;
        let identity = domain_identity(
            CURRENT_RECORD_VERIFICATION_IDENTITY_DOMAIN,
            &verification[..identity_offset],
        );
        verification[identity_offset..].copy_from_slice(&identity);
        let attestation = rebuild_current_record_attestation(
            fixture,
            attestation.to_vec(),
            challenge,
            &verification,
        );
        (verification, attestation)
    }

    fn alternate_carriage_for_same_subject(
        fixture: &Fixture,
    ) -> CompilerExecutionReceiptCarriageV1 {
        let challenge = CompilerExecutionAttestationChallengeV1::new(
            &fixture.policy,
            &fixture.subject,
            [0xa1; 32],
            2,
            fixture.carriage.acknowledgment().current_rollback_anchor(),
        )
        .unwrap();
        let request =
            CompilerExecutionAttestationRequestV1::new(challenge, fixture.subject.clone()).unwrap();
        let receipt = CompilerExecutionAttestationReceiptV1::issue(
            &fixture.policy,
            &request,
            &fixture.signing_key,
        )
        .unwrap();
        let publication =
            CompilerExecutionReceiptPublicationV1::new([0xa2; 32], [0xa3; 32], receipt).unwrap();
        let acknowledgment =
            CompilerExecutionReceiptPublicationAckV1::new(&publication, [0xa4; 32]).unwrap();
        CompilerExecutionReceiptCarriageV1::new(
            fixture.policy.clone(),
            request,
            publication,
            acknowledgment,
        )
        .unwrap()
    }

    #[test]
    fn signed_current_record_is_owned_once_without_authority() {
        let fixture = Fixture::new(0x20);
        let (client, service) = socket_pair();
        let service_carriage = fixture.carriage.clone();
        let service_policy = fixture.policy.clone();
        let service_key = fixture.signing_key.clone();
        let service_anchor_receipt = fixture.anchor_receipt();
        let service_anchor_key = fixture.anchor_signing_key.clone();
        let expected_challenge_bytes = [0xb7; 32];
        let service = thread::spawn(move || {
            let request = receive_request(&service);
            assert_eq!(
                request.kind(),
                CompilerExecutionServiceRequestKindV1::VerifyCurrent
            );
            assert_eq!(request.carriage(), Some(&service_carriage));
            let verification_challenge = request.verification_challenge().unwrap();
            assert_eq!(verification_challenge, expected_challenge_bytes);
            let currentness_challenge = CompilerExecutionCurrentRecordVerificationV3::external_anchor_currentness_challenge(
                &service_carriage,
                &service_anchor_receipt,
                verification_challenge,
            )
            .unwrap();
            let anchor_key =
                PinnedAnchorKeyV1::from_bytes(service_anchor_key.verifying_key().to_bytes())
                    .unwrap();
            let unsigned = UnsignedAnchorObservationV1::from_challenge(
                &currentness_challenge,
                AnchorPositionV1::Proposed,
            );
            let signature = service_anchor_key.sign(&unsigned.signing_bytes());
            let currentness_receipt = AnchorTransitionReceiptV1::new(
                currentness_challenge,
                &unsigned.attach_signature(signature.to_bytes()),
                &anchor_key,
            )
            .unwrap();
            let verification = CompilerExecutionCurrentRecordVerificationV3::new(
                &service_carriage,
                service_anchor_receipt,
                currentness_receipt,
                verification_challenge,
                [0x91; 32],
                [0x92; 32],
            )
            .unwrap();
            let attestation = CompilerExecutionCurrentRecordAttestationV3::issue(
                &service_policy,
                &service_carriage,
                verification,
                verification_challenge,
                &service_key,
            )
            .unwrap();
            let response = CompilerExecutionServiceResponseV1::verified_current(
                request.identity(),
                attestation,
            )
            .unwrap();
            send_response(&service, response.canonical_bytes());
        });
        let mut auditor =
            InheritedWorkerV3CompilerCurrentRecordAuditorV1::admit_owned_application_service(
                client,
            )
            .unwrap();
        let evidence = auditor
            .audit_exact_with_challenge(
                &fixture.subject,
                &fixture.carriage,
                CompilerExecutionCurrentRecordChallengeV1::from_bytes(expected_challenge_bytes)
                    .unwrap(),
            )
            .unwrap();
        let (verification_bytes, attestation_bytes, verification_identity, attestation_identity) = {
            let view = evidence.canonical_evidence_view();
            let decoded_verification = CompilerExecutionCurrentRecordVerificationV3::decode(
                view.verification_canonical_bytes(),
            )
            .unwrap();
            let decoded_attestation = CompilerExecutionCurrentRecordAttestationV3::decode(
                view.attestation_canonical_bytes(),
            )
            .unwrap();
            assert_eq!(
                decoded_verification.identity(),
                view.verification_identity()
            );
            assert_eq!(decoded_attestation.identity(), view.attestation_identity());
            assert_eq!(view.verification_challenge(), expected_challenge_bytes);
            assert_eq!(
                decoded_attestation.verification().identity(),
                view.verification_identity()
            );
            assert_eq!(
                decoded_attestation.verification().canonical_bytes(),
                view.verification_canonical_bytes()
            );
            assert!(!view.grants_verification_authority());
            assert!(!view.grants_authority());
            assert!(!view.grants_load_authority());
            assert!(!view.grants_launch_authority());
            (
                *view.verification_canonical_bytes(),
                *view.attestation_canonical_bytes(),
                view.verification_identity(),
                view.attestation_identity(),
            )
        };
        let mut mutated_verification = verification_bytes;
        *mutated_verification.last_mut().unwrap() ^= 1;
        assert!(
            CompilerExecutionCurrentRecordVerificationV3::decode(&mutated_verification).is_err()
        );
        let mut mutated_attestation = attestation_bytes;
        *mutated_attestation.last_mut().unwrap() ^= 1;
        assert!(CompilerExecutionCurrentRecordAttestationV3::decode(&mutated_attestation).is_err());
        assert!(evidence.authenticates_pinned_signing_key());
        assert!(evidence.authenticates_expected_fresh_challenge());
        assert_eq!(
            evidence
                .verification()
                .protected_policy_verification_identity(),
            [0x91; 32]
        );
        assert_eq!(
            evidence
                .verification()
                .protected_worker_ledger_verification_identity(),
            [0x92; 32]
        );
        assert_ne!(evidence.attestation_identity().as_bytes(), &[0; 32]);
        assert!(!evidence.authenticates_protected_key_custody());
        assert!(!evidence.authenticates_protected_current_record());
        assert!(evidence.authenticates_external_anchor_commit());
        assert_ne!(evidence.external_rollback_verification_identity(), [0; 32]);
        assert!(evidence.authenticates_external_rollback_currentness());
        assert!(!evidence.grants_verification_authority());
        assert!(!evidence.grants_authority());
        assert!(!evidence.grants_load_authority());
        assert!(!evidence.grants_launch_authority());
        assert!(matches!(
            auditor.audit_exact(&fixture.subject, &fixture.carriage),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::AlreadyConsumed)
        ));
        let bound = evidence
            .bind_exact_compiler_execution_v1(&fixture.subject, &fixture.carriage)
            .unwrap();
        assert_eq!(bound.subject_sha256(), *fixture.subject.identity().sha256());
        assert_eq!(
            bound.carriage_sha256(),
            *fixture.carriage.identity().as_bytes()
        );
        assert_ne!(bound.current_record_verification_sha256(), [0; 32]);
        assert_ne!(bound.current_record_attestation_sha256(), [0; 32]);
        let bound_view = bound.current_record_evidence_view().unwrap();
        assert_eq!(
            bound_view.verification_canonical_bytes(),
            &verification_bytes
        );
        assert_eq!(bound_view.attestation_canonical_bytes(), &attestation_bytes);
        assert_eq!(bound_view.verification_identity(), verification_identity);
        assert_eq!(bound_view.attestation_identity(), attestation_identity);
        assert_eq!(
            bound.current_record_verification_sha256(),
            *bound_view.verification_identity().as_bytes()
        );
        assert_eq!(
            bound.current_record_attestation_sha256(),
            *bound_view.attestation_identity().as_bytes()
        );
        assert!(bound.authenticates_signed_currentness_evidence());
        assert!(!bound.grants_verification_authority());
        service.join().unwrap();
    }

    #[test]
    fn subject_substitution_and_closed_service_fail_closed_and_consume_once() {
        let fixture = Fixture::new(0x20);
        let substituted = subject(0x21);
        let (client, _service) = socket_pair();
        let client = CompilerExecutionClientV1::admit(client, Duration::from_secs(2)).unwrap();
        let mut auditor = InheritedWorkerV3CompilerCurrentRecordAuditorV1::from_client(client);
        assert!(matches!(
            auditor.audit_exact(&substituted, &fixture.carriage),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::RequestMismatch)
        ));
        assert!(matches!(
            auditor.audit_exact(&fixture.subject, &fixture.carriage),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::AlreadyConsumed)
        ));

        let (client, service) = socket_pair();
        drop(service);
        let client = CompilerExecutionClientV1::admit(client, Duration::from_secs(2)).unwrap();
        let mut auditor = InheritedWorkerV3CompilerCurrentRecordAuditorV1::from_client(client);
        assert!(matches!(
            auditor.audit_exact(&fixture.subject, &fixture.carriage),
            Err(WorkerV3CompilerCurrentRecordAuditErrorV1::Client(_))
        ));
    }

    #[test]
    fn move_only_current_record_join_rejects_subject_and_carriage_substitution() {
        let fixture = Fixture::new(0x20);
        let substituted_subject = subject(0x21);
        assert!(matches!(
            direct_current_record_audit(&fixture, [0xb1; 32])
                .bind_exact_compiler_execution_v1(&substituted_subject, &fixture.carriage),
            Err(WorkerV3CompilerExecutionEvidenceErrorV1::RequestMismatch)
        ));

        let substituted_carriage = alternate_carriage_for_same_subject(&fixture);
        assert!(matches!(
            direct_current_record_audit(&fixture, [0xb2; 32])
                .bind_exact_compiler_execution_v1(&fixture.subject, &substituted_carriage),
            Err(WorkerV3CompilerExecutionEvidenceErrorV1::IdentityMismatch(
                "compiler-execution carriage"
            ))
        ));
    }

    #[test]
    fn canonical_current_record_bridge_admits_exact_evidence_without_authority() {
        let fixture = Fixture::new(0x20);
        let challenge = [0xc1; 32];
        let (verification, attestation) = canonical_current_record_evidence(&fixture, challenge);
        let admitted = admit_worker_v3_compiler_current_record_evidence_v1(
            &fixture.policy,
            &fixture.subject,
            &fixture.carriage,
            challenge,
            &verification,
            &attestation,
        )
        .unwrap();
        assert_eq!(
            admitted.current_record_verification_sha256(),
            *CompilerExecutionCurrentRecordVerificationV3::decode(&verification)
                .unwrap()
                .identity()
                .as_bytes()
        );
        assert!(admitted.authenticates_signed_currentness_evidence());
        assert!(!admitted.authenticates_protected_current_record());
        assert!(!admitted.grants_verification_authority());
    }

    #[test]
    fn canonical_current_record_bridge_rejects_challenge_policy_subject_and_carriage_mismatch() {
        let fixture = Fixture::new(0x20);
        let challenge = [0xc2; 32];
        let (verification, attestation) = canonical_current_record_evidence(&fixture, challenge);

        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &fixture.subject,
                &fixture.carriage,
                [0xc3; 32],
                &verification,
                &attestation,
            ),
            Err(
                WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation(
                    CompilerExecutionCurrentRecordVerificationErrorV3::ChallengeMismatch
                )
            )
        ));

        let alternate_policy = CompilerExecutionIssuerPolicyV1::new(
            8,
            CompilerExecutionIssuerMeasurementV1::new([0x71; 32], 789).unwrap(),
            CompilerExecutionIssuerMeasurementV1::new([0x72; 32], 987).unwrap(),
            SigningKey::from_bytes(&[0x73; 32])
                .verifying_key()
                .to_bytes(),
            SigningKey::from_bytes(&[0x74; 32])
                .verifying_key()
                .to_bytes(),
        )
        .unwrap();
        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &alternate_policy,
                &fixture.subject,
                &fixture.carriage,
                challenge,
                &verification,
                &attestation,
            ),
            Err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::PolicyMismatch)
        ));

        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &subject(0x21),
                &fixture.carriage,
                challenge,
                &verification,
                &attestation,
            ),
            Err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::RequestMismatch)
        ));

        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &fixture.subject,
                &alternate_carriage_for_same_subject(&fixture),
                challenge,
                &verification,
                &attestation,
            ),
            Err(
                WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation(
                    CompilerExecutionCurrentRecordVerificationErrorV3::VerificationMismatch
                )
            )
        ));
    }

    #[test]
    fn canonical_current_record_bridge_rejects_ledger_and_rollback_substitution() {
        let fixture = Fixture::new(0x20);
        let challenge = [0xc4; 32];
        let (verification, attestation) = canonical_current_record_evidence(&fixture, challenge);

        let (ledger_verification, ledger_attestation) = mutate_verification_identity_field(
            &fixture,
            challenge,
            &verification,
            &attestation,
            CURRENT_RECORD_WORKER_LEDGER_OFFSET,
            [0xd1; 32],
        );
        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &fixture.subject,
                &fixture.carriage,
                challenge,
                &ledger_verification,
                &ledger_attestation,
            ),
            Err(
                WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation(
                    CompilerExecutionCurrentRecordVerificationErrorV3::VerificationMismatch
                )
            )
        ));

        let (rollback_verification, rollback_attestation) = mutate_verification_identity_field(
            &fixture,
            challenge,
            &verification,
            &attestation,
            CURRENT_RECORD_CURRENT_ROLLBACK_ANCHOR_OFFSET,
            [0xd2; 32],
        );
        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &fixture.subject,
                &fixture.carriage,
                challenge,
                &rollback_verification,
                &rollback_attestation,
            ),
            Err(
                WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation(
                    CompilerExecutionCurrentRecordVerificationErrorV3::VerificationMismatch
                )
            )
        ));
    }

    #[test]
    fn canonical_current_record_bridge_rejects_stale_external_currentness() {
        let fixture = Fixture::new(0x20);
        let stale_challenge = [0xc5; 32];
        let expected_challenge = [0xc6; 32];
        let (verification, attestation) =
            canonical_current_record_evidence(&fixture, stale_challenge);
        let attestation = rebuild_current_record_attestation(
            &fixture,
            attestation,
            expected_challenge,
            &verification,
        );

        assert!(matches!(
            admit_worker_v3_compiler_current_record_evidence_v1(
                &fixture.policy,
                &fixture.subject,
                &fixture.carriage,
                expected_challenge,
                &verification,
                &attestation,
            ),
            Err(WorkerV3CompilerCurrentRecordEvidenceAdmissionErrorV1::Attestation(
                CompilerExecutionCurrentRecordVerificationErrorV3::ExternalAnchorCurrentnessReceiptMismatch
            ))
        ));
    }

    fn socket_pair() -> (OwnedFd, OwnedFd) {
        let mut descriptors = [-1_i32; 2];
        assert_eq!(
            unsafe {
                libc::socketpair(
                    libc::AF_UNIX,
                    libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                    0,
                    descriptors.as_mut_ptr(),
                )
            },
            0
        );
        unsafe {
            (
                OwnedFd::from_raw_fd(descriptors[0]),
                OwnedFd::from_raw_fd(descriptors[1]),
            )
        }
    }

    fn receive_request(service: &OwnedFd) -> CompilerExecutionServiceRequestV1 {
        let mut bytes = [0_u8; MAX_COMPILER_EXECUTION_SERVICE_REQUEST_BYTES_V1];
        let received = unsafe {
            libc::recv(
                service.as_raw_fd(),
                bytes.as_mut_ptr().cast(),
                bytes.len(),
                0,
            )
        };
        assert!(received > 0);
        CompilerExecutionServiceRequestV1::decode(&bytes[..received as usize]).unwrap()
    }

    fn send_response(service: &OwnedFd, bytes: &[u8]) {
        let sent = unsafe {
            libc::send(
                service.as_raw_fd(),
                bytes.as_ptr().cast(),
                bytes.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        assert_eq!(sent, bytes.len() as isize);
    }

    fn subject(seed: u8) -> InertCompilerExecutionSubjectV1 {
        let closure_pins = [
            [seed; 32],
            [seed + 1; 32],
            [seed + 2; 32],
            [seed + 3; 32],
            [seed + 4; 32],
            [seed + 5; 32],
        ];
        let mut closure_digest = Sha256::new();
        closure_digest.update(COMPILER_CLOSURE_IDENTITY_DOMAIN);
        closure_digest.update(1_u16.to_le_bytes());
        for pin in closure_pins {
            closure_digest.update(pin);
        }
        let closure_identity: [u8; 32] = closure_digest.finalize().into();
        let mut bytes = [0_u8; INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V1];
        let mut offset = 0;
        put(
            &mut bytes,
            &mut offset,
            &INERT_COMPILER_EXECUTION_SUBJECT_MAGIC_V1,
        );
        put(
            &mut bytes,
            &mut offset,
            &INERT_COMPILER_EXECUTION_SUBJECT_VERSION_V1.to_le_bytes(),
        );
        put(&mut bytes, &mut offset, &0_u16.to_le_bytes());
        put(
            &mut bytes,
            &mut offset,
            &(INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V1 as u64).to_le_bytes(),
        );
        put(&mut bytes, &mut offset, &0_u32.to_le_bytes());
        put(&mut bytes, &mut offset, &9_u64.to_le_bytes());
        put(&mut bytes, &mut offset, &[seed + 6; 16]);
        put(&mut bytes, &mut offset, &[seed + 7; 32]);
        bytes[offset] = 0;
        offset += 8;
        put(&mut bytes, &mut offset, &[seed + 8; 32]);
        put(&mut bytes, &mut offset, &[seed + 9; 32]);
        for pin in closure_pins {
            put(&mut bytes, &mut offset, &pin);
        }
        put(&mut bytes, &mut offset, &1_u16.to_le_bytes());
        put(&mut bytes, &mut offset, &closure_identity);
        for axis in 0_u8..7 {
            put(&mut bytes, &mut offset, &[seed + 10 + axis; 32]);
            put(
                &mut bytes,
                &mut offset,
                &(1_000_u64 + u64::from(axis)).to_le_bytes(),
            );
        }
        let identity = digest(SUBJECT_IDENTITY_DOMAIN, &bytes[..offset]);
        put(&mut bytes, &mut offset, &identity);
        assert_eq!(offset, bytes.len());
        InertCompilerExecutionSubjectV1::decode(&bytes).unwrap()
    }

    fn digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
        digest.finalize().into()
    }

    fn put(output: &mut [u8], offset: &mut usize, value: &[u8]) {
        let end = *offset + value.len();
        output[*offset..end].copy_from_slice(value);
        *offset = end;
    }
}
