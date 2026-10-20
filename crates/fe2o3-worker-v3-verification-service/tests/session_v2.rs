use std::collections::HashSet;
#[path = "support/readiness_v3.rs"]
mod readiness_v3_cases;
use std::fs::File;
use std::io::{IoSlice, IoSliceMut, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ed25519_dalek::{Signer, SigningKey};
use fe2o3_artifact_transaction::{
    INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V1, INERT_COMPILER_EXECUTION_SUBJECT_MAGIC_V1,
    INERT_COMPILER_EXECUTION_SUBJECT_VERSION_V1, InertCompilerExecutionSubjectV1,
};
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionAttestationChallengeV1, CompilerExecutionAttestationReceiptV1,
    CompilerExecutionAttestationRequestV1, CompilerExecutionCurrentRecordAttestationV3,
    CompilerExecutionCurrentRecordVerificationV3, CompilerExecutionExternalAnchorTransactionV1,
    CompilerExecutionIssuerMeasurementV1, CompilerExecutionIssuerPolicyV1,
    CompilerExecutionReceiptCarriageV1, CompilerExecutionReceiptPublicationAckV1,
    CompilerExecutionReceiptPublicationV1,
};
use fe2o3_external_anchor_protocol::{
    AnchorPositionV1, AnchorTransitionReceiptV1, AnchoredStateV1, CallerNonceV1, HashChainHeadV1,
    PinnedAnchorKeyV1, UnsignedAnchorObservationV1,
};
use fe2o3_worker_v3_verification_client::{
    WorkerV3VerificationBeginOutcomeV2 as ClientBeginOutcomeV2, WorkerV3VerificationClientErrorV2,
    WorkerV3VerificationClientV2, WorkerV3VerificationClientV3,
    WorkerV3VerificationPayloadSnapshotsV1,
};
use fe2o3_worker_v3_verification_protocol::{
    MAX_WORKER_V3_VERIFICATION_APPLICATION_RESPONSE_BYTES_V2,
    WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2, WorkerV3VerificationChallengeFrameV2,
    WorkerV3VerificationChallengeReservationV2, WorkerV3VerificationCurrentRecordFrameV2,
    WorkerV3VerificationEntryCoordinateV1, WorkerV3VerificationFdPayloadDescriptorV1,
    WorkerV3VerificationFdPayloadKindV1, WorkerV3VerificationFreshChallengeV1,
    WorkerV3VerificationMeasurementIdentityV1, WorkerV3VerificationPolicyIdentityV1,
    WorkerV3VerificationRequestV1, WorkerV3VerificationRosterIdentityV1,
    WorkerV3VerificationTerminalDispositionV2, WorkerV3VerificationTerminalFrameV2,
};
use fe2o3_worker_v3_verification_service::{
    WorkerV3VerificationAcceptedServiceEndpointV2, WorkerV3VerificationBeginOutcomeV2,
    WorkerV3VerificationCallerV1, WorkerV3VerificationChallengeReplayGuardV1,
    WorkerV3VerificationChallengeReservationProviderV2, WorkerV3VerificationCurrentRecordOutcomeV2,
    WorkerV3VerificationMeasurementResolverV1, WorkerV3VerificationPolicyResolverV1,
    WorkerV3VerificationReadyServiceSessionV3, WorkerV3VerificationRejectionReasonV1,
    WorkerV3VerificationRejectionReasonV2, WorkerV3VerificationServiceErrorV1,
    WorkerV3VerificationServiceErrorV2, begin_worker_v3_verification_session_until_v2,
    begin_worker_v3_verification_session_v2, prepare_worker_v3_verification_receiver_v1,
};
use rustix::fs::{MemfdFlags, Mode, OFlags, SealFlags};
use rustix::io::Errno;
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, Shutdown, SocketAddrUnix, SocketFlags, SocketType,
    accept_with, bind, connect, listen, recv, recvmsg, send, sendmsg, shutdown, socket_with,
    socketpair,
};
use sha2::{Digest, Sha256};

const SUBJECT_IDENTITY_DOMAIN: &[u8] = b"FE2O3/INERT-COMPILER-EXECUTION-SUBJECT/V1\0";
const COMPILER_CLOSURE_IDENTITY_DOMAIN: &[u8] = b"fe2o3-compiler-closure-identity-v2\0";
const ENVELOPE: &[u8] = b"canonical-v2-load-envelope";
const HSACO: &[u8] = b"canonical-finalized-hsaco";
const CROSS_PHASE_TIMEOUT: Duration = Duration::from_secs(1);
const EXPIRED_PHASE_DELAY: Duration = Duration::from_millis(1_100);
const SEALS: SealFlags = SealFlags::WRITE
    .union(SealFlags::GROW)
    .union(SealFlags::SHRINK)
    .union(SealFlags::SEAL);

#[repr(align(16))]
struct AlignedAncillaryStorage<const N: usize>([MaybeUninit<u8>; N]);

struct FixedPolicy(Option<WorkerV3VerificationPolicyIdentityV1>);

impl WorkerV3VerificationPolicyResolverV1 for FixedPolicy {
    fn resolve_expected_policy(
        &mut self,
        _caller: WorkerV3VerificationCallerV1,
        _request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationPolicyIdentityV1> {
        self.0
    }
}

struct FixedMeasurement(Option<WorkerV3VerificationMeasurementIdentityV1>);

impl WorkerV3VerificationMeasurementResolverV1 for FixedMeasurement {
    fn resolve_expected_measurement(
        &mut self,
        _caller: WorkerV3VerificationCallerV1,
        _policy: WorkerV3VerificationPolicyIdentityV1,
        _request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationMeasurementIdentityV1> {
        self.0
    }
}

type ReplayKey = (u32, u32, u32, [u8; 32], [u8; 32]);

#[derive(Default)]
struct ReplayGuard(HashSet<ReplayKey>);

impl WorkerV3VerificationChallengeReplayGuardV1 for ReplayGuard {
    fn admit_fresh_challenge(
        &mut self,
        caller: WorkerV3VerificationCallerV1,
        policy: WorkerV3VerificationPolicyIdentityV1,
        challenge: WorkerV3VerificationFreshChallengeV1,
    ) -> bool {
        self.0.insert((
            caller.pid(),
            caller.uid(),
            caller.gid(),
            *policy.as_bytes(),
            *challenge.as_bytes(),
        ))
    }
}

struct FixedReservations {
    challenge: [u8; 32],
    identity: [u8; 32],
    available: bool,
}

impl FixedReservations {
    fn available(challenge: u8, identity: u8) -> Self {
        Self {
            challenge: [challenge; 32],
            identity: [identity; 32],
            available: true,
        }
    }
}

impl WorkerV3VerificationChallengeReservationProviderV2 for FixedReservations {
    fn reserve_current_record_challenge(
        &mut self,
        _caller: WorkerV3VerificationCallerV1,
        _request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationChallengeReservationV2> {
        self.available.then(|| {
            WorkerV3VerificationChallengeReservationV2::new(self.challenge, self.identity).unwrap()
        })
    }
}

struct PausingReservations {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    reservation: Option<WorkerV3VerificationChallengeReservationV2>,
}

impl WorkerV3VerificationChallengeReservationProviderV2 for PausingReservations {
    fn reserve_current_record_challenge(
        &mut self,
        _caller: WorkerV3VerificationCallerV1,
        _request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationChallengeReservationV2> {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(2)).unwrap();
        self.reservation.take()
    }
}

#[derive(Clone)]
struct CurrentRecordFixture {
    signing_key: SigningKey,
    anchor_signing_key: SigningKey,
    policy: CompilerExecutionIssuerPolicyV1,
    carriage: CompilerExecutionReceiptCarriageV1,
    publication_request: CompilerExecutionAttestationRequestV1,
    publication: CompilerExecutionReceiptPublicationV1,
}

impl CurrentRecordFixture {
    fn new() -> Self {
        let signing_key = SigningKey::from_bytes(&[0x51; 32]);
        let anchor_signing_key = SigningKey::from_bytes(&[0x52; 32]);
        let policy = CompilerExecutionIssuerPolicyV1::new(
            7,
            CompilerExecutionIssuerMeasurementV1::new([0x61; 32], 123).unwrap(),
            CompilerExecutionIssuerMeasurementV1::new([0x62; 32], 456).unwrap(),
            signing_key.verifying_key().to_bytes(),
            anchor_signing_key.verifying_key().to_bytes(),
        )
        .unwrap();
        let subject = subject(0x20);
        let challenge =
            CompilerExecutionAttestationChallengeV1::new(&policy, &subject, [0x63; 32], 1, [0; 32])
                .unwrap();
        let publication_request =
            CompilerExecutionAttestationRequestV1::new(challenge, subject).unwrap();
        let receipt = CompilerExecutionAttestationReceiptV1::issue(
            &policy,
            &publication_request,
            &signing_key,
        )
        .unwrap();
        let publication =
            CompilerExecutionReceiptPublicationV1::new([0x64; 32], [0x65; 32], receipt).unwrap();
        let acknowledgment =
            CompilerExecutionReceiptPublicationAckV1::new(&publication, [0x66; 32]).unwrap();
        let carriage = CompilerExecutionReceiptCarriageV1::new(
            policy.clone(),
            publication_request.clone(),
            publication.clone(),
            acknowledgment,
        )
        .unwrap();
        Self {
            signing_key,
            anchor_signing_key,
            policy,
            carriage,
            publication_request,
            publication,
        }
    }

    fn records(
        &self,
        challenge: [u8; 32],
    ) -> (
        CompilerExecutionCurrentRecordVerificationV3,
        CompilerExecutionCurrentRecordAttestationV3,
    ) {
        let transaction = CompilerExecutionExternalAnchorTransactionV1::new(
            self.policy.clone(),
            self.publication_request.clone(),
            self.publication.clone(),
        )
        .unwrap();
        let anchor_key =
            PinnedAnchorKeyV1::from_bytes(self.anchor_signing_key.verifying_key().to_bytes())
                .unwrap();
        let pending = AnchoredStateV1::from_local_state(0, HashChainHeadV1::from_bytes([0; 32]))
            .prepare(transaction.external_anchor_digest(), &anchor_key)
            .unwrap()
            .begin_advance(CallerNonceV1::from_bytes([0x67; 32]), &anchor_key)
            .unwrap();
        let unsigned = UnsignedAnchorObservationV1::from_challenge(
            pending.challenge(),
            AnchorPositionV1::Proposed,
        );
        let signature = self
            .anchor_signing_key
            .sign(&unsigned.signing_bytes())
            .to_bytes();
        let commit_receipt = AnchorTransitionReceiptV1::new(
            pending.challenge().clone(),
            &unsigned.attach_signature(signature),
            &anchor_key,
        )
        .unwrap();
        let currentness_challenge =
            CompilerExecutionCurrentRecordVerificationV3::external_anchor_currentness_challenge(
                &self.carriage,
                &commit_receipt,
                challenge,
            )
            .unwrap();
        let unsigned = UnsignedAnchorObservationV1::from_challenge(
            &currentness_challenge,
            AnchorPositionV1::Proposed,
        );
        let signature = self
            .anchor_signing_key
            .sign(&unsigned.signing_bytes())
            .to_bytes();
        let currentness_receipt = AnchorTransitionReceiptV1::new(
            currentness_challenge,
            &unsigned.attach_signature(signature),
            &anchor_key,
        )
        .unwrap();
        let verification = CompilerExecutionCurrentRecordVerificationV3::new(
            &self.carriage,
            commit_receipt,
            currentness_receipt,
            challenge,
            [0x91; 32],
            [0x92; 32],
        )
        .unwrap();
        let attestation = CompilerExecutionCurrentRecordAttestationV3::issue(
            &self.policy,
            &self.carriage,
            verification.clone(),
            challenge,
            &self.signing_key,
        )
        .unwrap();
        (verification, attestation)
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn verification_request(challenge: u8) -> WorkerV3VerificationRequestV1 {
    WorkerV3VerificationRequestV1::new(
        WorkerV3VerificationFreshChallengeV1::new([challenge; 32]).unwrap(),
        WorkerV3VerificationRosterIdentityV1::new([0x22; 32]).unwrap(),
        WorkerV3VerificationPolicyIdentityV1::new([0x23; 32]).unwrap(),
        WorkerV3VerificationMeasurementIdentityV1::new([0x24; 32]).unwrap(),
        WorkerV3VerificationFdPayloadDescriptorV1::load_envelope_v2(
            ENVELOPE.len() as u64,
            sha256(ENVELOPE),
        )
        .unwrap(),
        WorkerV3VerificationFdPayloadDescriptorV1::finalized_hsaco(
            HSACO.len() as u64,
            sha256(HSACO),
        )
        .unwrap(),
        vec![
            WorkerV3VerificationEntryCoordinateV1::new(
                0,
                "kernel",
                "kernel_export",
                [0x31; 32],
                [0x32; 32],
                [0x33; 32],
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn sealed_read_only(bytes: &[u8]) -> File {
    let descriptor = rustix::fs::memfd_create(
        "fe2o3-worker-v3-v2-test",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .unwrap();
    let mut writer = File::from(descriptor);
    rustix::fs::fchmod(&writer, Mode::RUSR).unwrap();
    writer.write_all(bytes).unwrap();
    writer.flush().unwrap();
    rustix::fs::fcntl_add_seals(&writer, SEALS).unwrap();
    let path = format!("/proc/self/fd/{}", writer.as_raw_fd());
    let retained = File::from(
        rustix::fs::open(path, OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty()).unwrap(),
    );
    drop(writer);
    retained
}

fn snapshots(request: &WorkerV3VerificationRequestV1) -> WorkerV3VerificationPayloadSnapshotsV1 {
    WorkerV3VerificationPayloadSnapshotsV1::admit(
        request,
        vec![
            sealed_read_only(ENVELOPE).into(),
            sealed_read_only(HSACO).into(),
        ],
    )
    .unwrap()
}

fn pair() -> (OwnedFd, OwnedFd) {
    let pair = socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    prepare_worker_v3_verification_receiver_v1(&pair.0).unwrap();
    pair
}

fn prepared_path_listener() -> (tempfile::TempDir, PathBuf, OwnedFd) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("service.sock");
    let address = SocketAddrUnix::new(&path).unwrap();
    let listener = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    bind(&listener, &address).unwrap();
    prepare_worker_v3_verification_receiver_v1(&listener).unwrap();
    assert!(rustix::net::sockopt::socket_passcred(&listener).unwrap());
    listen(&listener, 1).unwrap();
    (root, path, listener)
}

fn admission_state(
    request: &WorkerV3VerificationRequestV1,
) -> (FixedPolicy, FixedMeasurement, ReplayGuard) {
    (
        FixedPolicy(Some(request.policy_identity())),
        FixedMeasurement(Some(request.measurement_identity())),
        ReplayGuard::default(),
    )
}

#[test]
fn exact_two_phase_session_returns_only_opaque_authority_free_application_bytes() {
    let request = verification_request(1);
    let fixture = CurrentRecordFixture::new();
    let (service, client) = pair();
    let client_request = request.clone();
    let client_fixture = fixture.clone();
    let client_thread = thread::spawn(move || {
        let deadline = Instant::now().checked_add(Duration::from_secs(2)).unwrap();
        let client = WorkerV3VerificationClientV2::admit_until(client, deadline).unwrap();
        assert_eq!(client.deadline(), deadline);
        let outcome = client
            .begin(client_request.clone(), snapshots(&client_request))
            .unwrap();
        let ClientBeginOutcomeV2::Reserved(begin) = outcome else {
            panic!("valid Begin was rejected");
        };
        let (challenge, pending) = begin.into_parts();
        assert_eq!(pending.deadline(), deadline);
        assert_eq!(challenge.reservation_identity(), &[0x82; 32]);
        let compiler_challenge = challenge.into_compiler_execution_challenge().unwrap();
        let challenge = *compiler_challenge.as_bytes();
        let (verification, attestation) = client_fixture.records(challenge);
        pending
            .submit_current_record(
                *verification.canonical_bytes(),
                *attestation.canonical_bytes(),
            )
            .unwrap()
    });
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x81, 0x82);
    let deadline = Instant::now().checked_add(Duration::from_secs(2)).unwrap();
    let begin = begin_worker_v3_verification_session_until_v2(
        service,
        deadline,
        &mut policy,
        &mut measurement,
        &mut replay,
        &mut reservations,
    )
    .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = begin else {
        panic!("valid Begin was rejected");
    };
    assert_eq!(pending.deadline(), deadline);
    assert_eq!(pending.reservation().challenge_bytes(), &[0x81; 32]);
    let current = pending.receive_current_record().unwrap();
    let WorkerV3VerificationCurrentRecordOutcomeV2::Ready(terminal) = current else {
        panic!("valid current record was rejected");
    };
    assert_eq!(terminal.deadline(), deadline);
    assert!(!terminal.grants_authority());
    assert_eq!(
        terminal
            .payload(WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2)
            .sha256(),
        &sha256(ENVELOPE)
    );
    let completed = terminal
        .send_application_response(b"opaque-application-decision".to_vec())
        .unwrap();
    assert!(completed.current_record().is_some());
    assert!(!completed.grants_authority());
    let terminal = client_thread.join().unwrap();
    assert_eq!(
        terminal.disposition(),
        WorkerV3VerificationTerminalDispositionV2::ApplicationResponse
    );
    assert_eq!(
        terminal.application_response_bytes(),
        b"opaque-application-decision"
    );
    assert!(!terminal.grants_authority());
}

#[test]
fn connected_path_v3_session_preserves_process_credentials_across_every_phase() {
    // Uncoordinated V2 pathname sessions require external rendezvous after both
    // strict checks. This positive instead uses V3's public readiness boundary.
    let request = verification_request(41);
    let (_root, path, listener) = prepared_path_listener();
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("cross_process_connected_path_v3_client_helper")
        .arg("--ignored")
        .env("FE2O3_WORKER_V3_CONNECTED_PATH_CLIENT", &path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let child_pid = child.id();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    assert!(rustix::net::sockopt::socket_passcred(&service).unwrap());
    let service = WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap();
    assert_eq!(service.caller().pid(), child_pid);

    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0xa1, 0xa2);
    let deadline = Instant::now().checked_add(Duration::from_secs(5)).unwrap();
    let ready =
        WorkerV3VerificationReadyServiceSessionV3::prepare_until(service, deadline).unwrap();
    assert_eq!(ready.caller().pid(), child_pid);
    assert_eq!(ready.deadline(), deadline);
    assert!(!ready.grants_authority());
    let begin = ready
        .begin(
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = begin else {
        panic!("valid connected-path Begin was rejected");
    };
    assert_eq!(pending.caller().pid(), child_pid);
    assert_eq!(pending.deadline(), deadline);
    assert!(!pending.grants_authority());
    let current = pending.receive_current_record().unwrap();
    let WorkerV3VerificationCurrentRecordOutcomeV2::Ready(terminal) = current else {
        panic!("valid connected-path current record was rejected");
    };
    terminal
        .send_application_response(b"connected-path-application-decision".to_vec())
        .unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "connected-path client failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "subprocess helper for the V3 connected-path session test"]
fn cross_process_connected_path_v3_client_helper() {
    let Some(path) = std::env::var_os("FE2O3_WORKER_V3_CONNECTED_PATH_CLIENT") else {
        return;
    };
    let path = PathBuf::from(path);
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    connect(&client, &SocketAddrUnix::new(&path).unwrap()).unwrap();
    let deadline = Instant::now().checked_add(Duration::from_secs(5)).unwrap();
    let client =
        WorkerV3VerificationClientV3::admit_connected_path_until(client, &path, deadline).unwrap();
    assert_eq!(client.deadline(), deadline);
    assert!(!client.grants_authority());
    let client = client.wait_ready().unwrap();
    assert_eq!(client.deadline(), deadline);
    assert!(!client.grants_authority());
    let request = verification_request(41);
    let outcome = client.begin(request.clone(), snapshots(&request)).unwrap();
    let ClientBeginOutcomeV2::Reserved(begin) = outcome else {
        panic!("valid connected-path Begin was rejected");
    };
    let (challenge, pending) = begin.into_parts();
    assert_eq!(challenge.reservation_identity(), &[0xa2; 32]);
    assert_eq!(challenge.as_bytes(), &[0xa1; 32]);
    assert_eq!(pending.deadline(), deadline);
    assert!(!pending.grants_authority());
    let fixture = CurrentRecordFixture::new();
    let (verification, attestation) = fixture.records(*challenge.as_bytes());
    let terminal = pending
        .submit_current_record(
            *verification.canonical_bytes(),
            *attestation.canonical_bytes(),
        )
        .unwrap();
    assert_eq!(
        terminal.disposition(),
        WorkerV3VerificationTerminalDispositionV2::ApplicationResponse
    );
    assert_eq!(
        terminal.application_response_bytes(),
        b"connected-path-application-decision"
    );
    assert!(!terminal.grants_authority());
}

#[test]
fn legacy_v2_connected_path_send_first_is_rejected_with_endpoint_retained() {
    let request = verification_request(42);
    let (_root, path, listener) = prepared_path_listener();
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    connect(&client, &SocketAddrUnix::new(&path).unwrap()).unwrap();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    let unnamed = rustix::net::SocketAddrAny::from(SocketAddrUnix::new_unnamed());
    assert_eq!(rustix::net::getpeername(&service).unwrap(), Some(unnamed));
    let sender = rustix::io::fcntl_dupfd_cloexec(&client, 0).unwrap();
    let client =
        WorkerV3VerificationClientV2::admit_connected_path(client, &path, Duration::from_secs(5))
            .unwrap();
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    // Complete the canonical Begin send before admission: SO_PASSCRED autobinds
    // the previously unnamed client. This is a V2 refusal, not a full positive.
    send_begin_raw(&sender, &request, &envelope, &hsaco);
    let peer = rustix::net::getpeername(&service)
        .unwrap()
        .expect("connected service endpoint must retain its peer address");
    assert!(
        SocketAddrUnix::try_from(peer)
            .unwrap()
            .abstract_name()
            .is_some()
    );
    let original = service.as_raw_fd();
    let failure = WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap_err();
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control is not the expected canonical filesystem-path service endpoint connected to an unnamed client"
        )
    ));
    let retained = failure.into_control();
    assert_eq!(retained.as_raw_fd(), original);
    rustix::io::fcntl_getfd(&retained).unwrap();
    drop(retained);
    drop(client);
    drop(sender);
    drop(envelope);
    drop(hsaco);
}

#[test]
fn accepted_path_admission_rejects_unnamed_abstract_and_named_client_substitutions() {
    let (service, _client) = pair();
    let original = service.as_raw_fd();
    let failure = WorkerV3VerificationAcceptedServiceEndpointV2::admit(
        service,
        Path::new("/run/fe2o3/worker-v3-verifier.sock"),
    )
    .unwrap_err();
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV1::InvalidControl(_)
    ));
    let retained = failure.into_control();
    assert_eq!(retained.as_raw_fd(), original);
    rustix::io::fcntl_getfd(&retained).unwrap();

    let (_root, path, listener) = prepared_path_listener();
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    connect(&client, &SocketAddrUnix::new(&path).unwrap()).unwrap();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    let wrong_path = path.with_file_name("other.sock");
    let failure =
        WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &wrong_path).unwrap_err();
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV1::InvalidControl(_)
    ));

    let (_root, path, listener) = prepared_path_listener();
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    let client_path = path.with_file_name("client.sock");
    bind(&client, &SocketAddrUnix::new(client_path).unwrap()).unwrap();
    connect(&client, &SocketAddrUnix::new(&path).unwrap()).unwrap();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    let client_failure =
        WorkerV3VerificationClientV2::admit_connected_path(client, &path, Duration::from_secs(1))
            .unwrap_err();
    let service_failure =
        WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap_err();
    assert!(matches!(
        client_failure.source_error(),
        WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer
    ));
    assert!(matches!(
        service_failure.source_error(),
        WorkerV3VerificationServiceErrorV1::InvalidControl(_)
    ));

    let abstract_name = format!("fe2o3-worker-v3-path-test-{}", std::process::id());
    let address = SocketAddrUnix::new_abstract_name(abstract_name.as_bytes()).unwrap();
    let listener = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    bind(&listener, &address).unwrap();
    prepare_worker_v3_verification_receiver_v1(&listener).unwrap();
    assert!(rustix::net::sockopt::socket_passcred(&listener).unwrap());
    listen(&listener, 1).unwrap();
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    connect(&client, &address).unwrap();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    let client_failure = WorkerV3VerificationClientV2::admit_connected_path(
        client,
        Path::new("/run/fe2o3/worker-v3-verifier.sock"),
        Duration::from_secs(1),
    )
    .unwrap_err();
    let service_failure = WorkerV3VerificationAcceptedServiceEndpointV2::admit(
        service,
        Path::new("/run/fe2o3/worker-v3-verifier.sock"),
    )
    .unwrap_err();
    assert!(matches!(
        client_failure.source_error(),
        WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer
    ));
    assert!(matches!(
        service_failure.source_error(),
        WorkerV3VerificationServiceErrorV1::InvalidControl(_)
    ));
}

#[test]
fn preexpired_absolute_deadlines_fail_before_any_descriptor_transfer() {
    let (service, client) = pair();
    assert!(matches!(
        WorkerV3VerificationClientV2::admit_until(client, Instant::now()),
        Err(WorkerV3VerificationClientErrorV2::Timeout)
    ));
    assert_eq!(
        recv(&service, &mut [0_u8; 1], RecvFlags::DONTWAIT)
            .unwrap()
            .0,
        0
    );

    let request = verification_request(36);
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    let retained_service = rustix::io::fcntl_dupfd_cloexec(&service, 0).unwrap();
    send_begin_raw(&peer, &request, &envelope, &hsaco);
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x9d, 0x9e);
    assert!(matches!(
        begin_worker_v3_verification_session_until_v2(
            service,
            Instant::now(),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        ),
        Err(WorkerV3VerificationServiceErrorV2::V1(
            WorkerV3VerificationServiceErrorV1::Timeout
        ))
    ));
    assert!(replay.0.is_empty());
    assert!(reservations.available);

    let expected = request.encode_canonical();
    let mut bytes = vec![0_u8; expected.len()];
    let mut space = AlignedAncillaryStorage(
        [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2), ScmCredentials(1))],
    );
    let mut ancillary = RecvAncillaryBuffer::new(&mut space.0);
    let received = {
        let mut vectors = [IoSliceMut::new(&mut bytes)];
        recvmsg(
            &retained_service,
            &mut vectors,
            &mut ancillary,
            RecvFlags::CMSG_CLOEXEC | RecvFlags::TRUNC,
        )
        .unwrap()
    };
    assert_eq!(received.bytes, expected.len());
    assert_eq!(bytes.as_slice(), expected);
    let mut transferred_descriptors = Vec::new();
    for message in ancillary.drain() {
        if let RecvAncillaryMessage::ScmRights(descriptors) = message {
            transferred_descriptors.extend(descriptors);
        }
    }
    assert_eq!(transferred_descriptors.len(), 2);
}

#[test]
fn unavailable_reservation_and_replayed_begin_fail_closed() {
    let request = verification_request(2);
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let mut replay = ReplayGuard::default();
    for expected in [
        WorkerV3VerificationRejectionReasonV2::ChallengeReservationUnavailable,
        WorkerV3VerificationRejectionReasonV2::Begin(
            WorkerV3VerificationRejectionReasonV1::ChallengeReplay,
        ),
    ] {
        let (service, peer) = pair();
        send_begin_raw(&peer, &request, &envelope, &hsaco);
        let mut policy = FixedPolicy(Some(request.policy_identity()));
        let mut measurement = FixedMeasurement(Some(request.measurement_identity()));
        let mut reservations = FixedReservations {
            challenge: [0x83; 32],
            identity: [0x84; 32],
            available: false,
        };
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(1),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Rejected(rejected) = outcome else {
            panic!("fail-closed Begin was reserved");
        };
        assert_eq!(rejected.reason(), expected);
        let response = receive_challenge(&peer);
        assert!(response.reservation().is_none());
        assert!(response.matches_request(&request));
        expect_peer_eof(&peer);
    }
}

#[test]
fn pipelined_packet_before_challenge_is_rejected_as_out_of_order() {
    let request = verification_request(3);
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    send_begin_raw(&peer, &request, &envelope, &hsaco);
    assert_eq!(send(&peer, b"early", SendFlags::NOSIGNAL).unwrap(), 5);
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x85, 0x86);
    let outcome = begin_worker_v3_verification_session_v2(
        service,
        Duration::from_secs(1),
        &mut policy,
        &mut measurement,
        &mut replay,
        &mut reservations,
    )
    .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Rejected(rejected) = outcome else {
        panic!("pipelined Begin was reserved");
    };
    assert_eq!(
        rejected.reason(),
        WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder
    );
    // Dropping a seqpacket endpoint with an unread out-of-order packet may make the peer observe
    // ECONNRESET instead of the already-sent generic rejection. Either way no reservation escapes.
}

#[test]
fn packet_queued_while_reserving_challenge_is_rejected_before_challenge_release() {
    let request = verification_request(30);
    let service_request = request.clone();
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    let (entered_tx, entered_rx) = mpsc::sync_channel(0);
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let service_thread = thread::spawn(move || {
        let (mut policy, mut measurement, mut replay) = admission_state(&service_request);
        let mut reservations = PausingReservations {
            entered: entered_tx,
            release: release_rx,
            reservation: Some(
                WorkerV3VerificationChallengeReservationV2::new([0x91; 32], [0x92; 32]).unwrap(),
            ),
        };
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(2),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Rejected(rejected) = outcome else {
            panic!("packet queued during reservation received a challenge");
        };
        assert!(rejected.frame().reservation().is_none());
        assert!(rejected.frame().matches_request(rejected.request()));
        rejected.reason()
    });

    send_begin_raw(&peer, &request, &envelope, &hsaco);
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(send(&peer, b"early", SendFlags::NOSIGNAL).unwrap(), 5);
    release_tx.send(()).unwrap();
    assert_eq!(
        service_thread.join().unwrap(),
        WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder
    );
    // The unread out-of-order packet can make the peer observe ECONNRESET instead of the rejection.
}

#[test]
fn ancillary_and_session_substitution_retain_custody_for_generic_rejection() {
    for ancillary in [true, false] {
        let request = verification_request(if ancillary { 4 } else { 5 });
        let fixture = CurrentRecordFixture::new();
        let envelope = sealed_read_only(ENVELOPE);
        let hsaco = sealed_read_only(HSACO);
        let (service, peer) = pair();
        send_begin_raw(&peer, &request, &envelope, &hsaco);
        let (mut policy, mut measurement, mut replay) = admission_state(&request);
        let mut reservations = FixedReservations::available(0x87, 0x88);
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(2),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
            panic!("valid Begin was rejected");
        };
        let challenge = receive_challenge(&peer).into_reservation().unwrap();
        let (verification, attestation) = fixture.records(*challenge.challenge_bytes());
        let frame_request = if ancillary {
            request.clone()
        } else {
            verification_request(99)
        };
        let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
            &frame_request,
            &challenge,
            verification.canonical_bytes(),
            attestation.canonical_bytes(),
        )
        .unwrap();
        if ancillary {
            send_current_with_descriptor(&peer, frame.encode_canonical(), &envelope);
        } else {
            assert_eq!(
                send(&peer, frame.encode_canonical(), SendFlags::NOSIGNAL).unwrap(),
                frame.encode_canonical().len()
            );
        }
        shutdown(&peer, Shutdown::Write).unwrap();
        let outcome = pending.receive_current_record().unwrap();
        let WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(rejected) = outcome else {
            panic!("hostile current record reached application state");
        };
        assert_eq!(
            rejected.reason(),
            if ancillary {
                WorkerV3VerificationRejectionReasonV2::CurrentRecordTransfer
            } else {
                WorkerV3VerificationRejectionReasonV2::CurrentRecordAssociation
            }
        );
        assert_eq!(
            rejected
                .payload(WorkerV3VerificationFdPayloadKindV1::FinalizedHsaco)
                .sha256(),
            &sha256(HSACO)
        );
        rejected.send_rejection().unwrap();
        let terminal = receive_terminal(&peer);
        assert_eq!(
            terminal.disposition(),
            WorkerV3VerificationTerminalDispositionV2::Rejected
        );
    }
}

#[test]
fn phase_two_endpoint_transfer_to_another_process_is_rejected_with_custody_retained() {
    let request = verification_request(31);
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    send_begin_raw(&peer, &request, &envelope, &hsaco);
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x93, 0x94);
    let outcome = begin_worker_v3_verification_session_v2(
        service,
        Duration::from_secs(10),
        &mut policy,
        &mut measurement,
        &mut replay,
        &mut reservations,
    )
    .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
        panic!("valid Begin was rejected");
    };
    let original_caller = pending.caller();
    let reservation = receive_challenge(&peer).into_reservation().unwrap();
    assert_eq!(reservation.challenge_bytes(), &[0x93; 32]);
    assert_eq!(reservation.reservation_identity(), &[0x94; 32]);

    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("cross_process_phase_two_sender_helper")
        .arg("--ignored")
        .env("FE2O3_WORKER_V3_PHASE_TWO_CHILD", "1")
        .stdin(Stdio::from(peer))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    assert_ne!(child.id(), original_caller.pid());

    let outcome = pending.receive_current_record().unwrap();
    let WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(rejected) = outcome else {
        panic!("current record from a different PID reached application state");
    };
    assert_eq!(
        rejected.reason(),
        WorkerV3VerificationRejectionReasonV2::CurrentRecordTransfer
    );
    assert_eq!(
        rejected
            .payload(WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2)
            .sha256(),
        &sha256(ENVELOPE)
    );
    assert_eq!(
        rejected
            .payload(WorkerV3VerificationFdPayloadKindV1::FinalizedHsaco)
            .sha256(),
        &sha256(HSACO)
    );
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "phase-two child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "subprocess helper for the cross-process credential test"]
fn cross_process_phase_two_sender_helper() {
    if std::env::var_os("FE2O3_WORKER_V3_PHASE_TWO_CHILD").is_none() {
        return;
    }
    let request = verification_request(31);
    let fixture = CurrentRecordFixture::new();
    let reservation =
        WorkerV3VerificationChallengeReservationV2::new([0x93; 32], [0x94; 32]).unwrap();
    let (verification, attestation) = fixture.records(*reservation.challenge_bytes());
    let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
        &request,
        &reservation,
        verification.canonical_bytes(),
        attestation.canonical_bytes(),
    )
    .unwrap();
    let stdin = std::io::stdin();
    assert_eq!(
        send(&stdin, frame.encode_canonical(), SendFlags::NOSIGNAL).unwrap(),
        frame.encode_canonical().len()
    );
    shutdown(&stdin, Shutdown::Write).unwrap();
}

#[test]
fn malformed_current_record_and_trailing_packet_or_missing_eof_fail_closed() {
    let cases = ["malformed", "trailing", "missing-eof"];
    for (index, case) in cases.into_iter().enumerate() {
        let request = verification_request(10 + index as u8);
        let fixture = CurrentRecordFixture::new();
        let envelope = sealed_read_only(ENVELOPE);
        let hsaco = sealed_read_only(HSACO);
        let (service, peer) = pair();
        send_begin_raw(&peer, &request, &envelope, &hsaco);
        let (mut policy, mut measurement, mut replay) = admission_state(&request);
        let mut reservations = FixedReservations::available(0x89, 0x8a);
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(1),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
            panic!("valid Begin was rejected");
        };
        let reservation = receive_challenge(&peer).into_reservation().unwrap();
        let (verification, attestation) = fixture.records(*reservation.challenge_bytes());
        let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
            &request,
            &reservation,
            verification.canonical_bytes(),
            attestation.canonical_bytes(),
        )
        .unwrap();
        let mut bytes = *frame.encode_canonical();
        if case == "malformed" {
            *bytes.last_mut().unwrap() ^= 1;
        }
        assert_eq!(
            send(&peer, &bytes, SendFlags::NOSIGNAL).unwrap(),
            bytes.len()
        );
        if case == "trailing" {
            assert_eq!(send(&peer, b"trailing", SendFlags::NOSIGNAL).unwrap(), 8);
        }
        if case != "missing-eof" {
            shutdown(&peer, Shutdown::Write).unwrap();
        }
        let outcome = pending.receive_current_record();
        match case {
            "malformed" => {
                let WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(rejected) =
                    outcome.unwrap()
                else {
                    panic!("malformed record reached application state");
                };
                assert_eq!(
                    rejected.reason(),
                    WorkerV3VerificationRejectionReasonV2::CurrentRecordFraming
                );
                rejected.send_rejection().unwrap();
                assert_eq!(
                    receive_terminal(&peer).disposition(),
                    WorkerV3VerificationTerminalDispositionV2::Rejected
                );
            }
            "trailing" => assert!(matches!(
                outcome,
                Err(WorkerV3VerificationServiceErrorV2::V1(
                    WorkerV3VerificationServiceErrorV1::TrailingTransfer
                ))
            )),
            "missing-eof" => assert!(matches!(
                outcome,
                Err(WorkerV3VerificationServiceErrorV2::V1(
                    WorkerV3VerificationServiceErrorV1::Timeout
                ))
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn service_deadline_expires_across_challenge_before_current_record() {
    let request = verification_request(32);
    let fixture = CurrentRecordFixture::new();
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    send_begin_raw(&peer, &request, &envelope, &hsaco);
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x95, 0x96);
    let outcome = begin_worker_v3_verification_session_v2(
        service,
        CROSS_PHASE_TIMEOUT,
        &mut policy,
        &mut measurement,
        &mut replay,
        &mut reservations,
    )
    .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
        panic!("valid Begin was rejected");
    };
    let reservation = receive_challenge(&peer).into_reservation().unwrap();
    let (verification, attestation) = fixture.records(*reservation.challenge_bytes());
    let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
        &request,
        &reservation,
        verification.canonical_bytes(),
        attestation.canonical_bytes(),
    )
    .unwrap();

    thread::sleep(EXPIRED_PHASE_DELAY);
    assert_eq!(
        send(&peer, frame.encode_canonical(), SendFlags::NOSIGNAL).unwrap(),
        frame.encode_canonical().len()
    );
    shutdown(&peer, Shutdown::Write).unwrap();
    assert!(matches!(
        pending.receive_current_record(),
        Err(WorkerV3VerificationServiceErrorV2::V1(
            WorkerV3VerificationServiceErrorV1::Timeout
        ))
    ));
}

#[test]
fn service_deadline_expires_before_terminal_send_and_retains_custody() {
    let (terminal, peer) = ready_terminal(verification_request(33), 0x97, 0x98);
    thread::sleep(EXPIRED_PHASE_DELAY);
    let failure = terminal
        .send_application_response(b"too-late".to_vec())
        .unwrap_err();
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV2::V1(WorkerV3VerificationServiceErrorV1::Timeout)
    ));
    let terminal = failure.into_session();
    assert_eq!(
        terminal
            .payload(WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2)
            .sha256(),
        &sha256(ENVELOPE)
    );
    drop(peer);
}

#[test]
fn client_deadline_expires_across_challenge_before_current_record() {
    let request = verification_request(34);
    let fixture = CurrentRecordFixture::new();
    let (service, client) = pair();
    let service_request = request.clone();
    let service_thread = thread::spawn(move || {
        let (mut policy, mut measurement, mut replay) = admission_state(&service_request);
        let mut reservations = FixedReservations::available(0x99, 0x9a);
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(3),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
            panic!("valid Begin was rejected");
        };
        let _ = pending.receive_current_record();
    });

    let outcome = WorkerV3VerificationClientV2::admit(client, CROSS_PHASE_TIMEOUT)
        .unwrap()
        .begin(request.clone(), snapshots(&request))
        .unwrap();
    let ClientBeginOutcomeV2::Reserved(begin) = outcome else {
        panic!("valid Begin was rejected");
    };
    let (challenge, pending) = begin.into_parts();
    let (verification, attestation) = fixture.records(*challenge.as_bytes());
    thread::sleep(EXPIRED_PHASE_DELAY);
    assert!(matches!(
        pending.submit_current_record(
            *verification.canonical_bytes(),
            *attestation.canonical_bytes(),
        ),
        Err(WorkerV3VerificationClientErrorV2::Timeout)
    ));
    service_thread.join().unwrap();
}

#[test]
fn client_deadline_expires_waiting_for_terminal_after_current_record() {
    let request = verification_request(35);
    let fixture = CurrentRecordFixture::new();
    let (service, client) = pair();
    let service_request = request.clone();
    let service_thread = thread::spawn(move || {
        let (mut policy, mut measurement, mut replay) = admission_state(&service_request);
        let mut reservations = FixedReservations::available(0x9b, 0x9c);
        let outcome = begin_worker_v3_verification_session_v2(
            service,
            Duration::from_secs(3),
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap();
        let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
            panic!("valid Begin was rejected");
        };
        let outcome = pending.receive_current_record().unwrap();
        let WorkerV3VerificationCurrentRecordOutcomeV2::Ready(terminal) = outcome else {
            panic!("valid current record was rejected");
        };
        thread::sleep(EXPIRED_PHASE_DELAY);
        let _ = terminal.send_application_response(b"too-late-for-client".to_vec());
    });

    let outcome = WorkerV3VerificationClientV2::admit(client, CROSS_PHASE_TIMEOUT)
        .unwrap()
        .begin(request.clone(), snapshots(&request))
        .unwrap();
    let ClientBeginOutcomeV2::Reserved(begin) = outcome else {
        panic!("valid Begin was rejected");
    };
    let (challenge, pending) = begin.into_parts();
    let (verification, attestation) = fixture.records(*challenge.as_bytes());
    assert!(matches!(
        pending.submit_current_record(
            *verification.canonical_bytes(),
            *attestation.canonical_bytes(),
        ),
        Err(WorkerV3VerificationClientErrorV2::Timeout)
    ));
    service_thread.join().unwrap();
}

#[test]
fn oversized_application_response_returns_the_ready_session_with_custody() {
    let (terminal, peer) = ready_terminal(verification_request(20), 0x8b, 0x8c);
    let failure = terminal
        .send_application_response(vec![
            0;
            MAX_WORKER_V3_VERIFICATION_APPLICATION_RESPONSE_BYTES_V2
                + 1
        ])
        .unwrap_err();
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV2::Protocol(_)
    ));
    let terminal = failure.into_session();
    assert_eq!(
        terminal
            .payload(WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2)
            .sha256(),
        &sha256(ENVELOPE)
    );
    terminal.send_rejection().unwrap();
    assert_eq!(
        receive_terminal(&peer).disposition(),
        WorkerV3VerificationTerminalDispositionV2::Rejected
    );
}

fn ready_terminal(
    request: WorkerV3VerificationRequestV1,
    challenge: u8,
    reservation_identity: u8,
) -> (
    fe2o3_worker_v3_verification_service::PendingWorkerV3VerificationTerminalSessionV2,
    OwnedFd,
) {
    let fixture = CurrentRecordFixture::new();
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let (service, peer) = pair();
    send_begin_raw(&peer, &request, &envelope, &hsaco);
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(challenge, reservation_identity);
    let outcome = begin_worker_v3_verification_session_v2(
        service,
        Duration::from_secs(1),
        &mut policy,
        &mut measurement,
        &mut replay,
        &mut reservations,
    )
    .unwrap();
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = outcome else {
        panic!("valid Begin was rejected");
    };
    let reservation = receive_challenge(&peer).into_reservation().unwrap();
    let (verification, attestation) = fixture.records(*reservation.challenge_bytes());
    let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
        &request,
        &reservation,
        verification.canonical_bytes(),
        attestation.canonical_bytes(),
    )
    .unwrap();
    assert_eq!(
        send(&peer, frame.encode_canonical(), SendFlags::NOSIGNAL).unwrap(),
        frame.encode_canonical().len()
    );
    shutdown(&peer, Shutdown::Write).unwrap();
    let WorkerV3VerificationCurrentRecordOutcomeV2::Ready(terminal) =
        pending.receive_current_record().unwrap()
    else {
        panic!("valid current record was rejected");
    };
    (terminal, peer)
}

fn send_begin_raw(
    peer: &OwnedFd,
    request: &WorkerV3VerificationRequestV1,
    envelope: &File,
    hsaco: &File,
) {
    let descriptors = [envelope.as_fd(), hsaco.as_fd()];
    let mut space =
        AlignedAncillaryStorage([MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2))]);
    let mut ancillary = SendAncillaryBuffer::new(&mut space.0);
    assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
    assert_eq!(
        sendmsg(
            peer,
            &[IoSlice::new(request.encode_canonical())],
            &mut ancillary,
            SendFlags::NOSIGNAL,
        )
        .unwrap(),
        request.encode_canonical().len()
    );
}

fn send_current_with_descriptor(peer: &OwnedFd, bytes: &[u8], descriptor: &File) {
    let descriptors = [descriptor.as_fd()];
    let mut space =
        AlignedAncillaryStorage([MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))]);
    let mut ancillary = SendAncillaryBuffer::new(&mut space.0);
    assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
    assert_eq!(
        sendmsg(
            peer,
            &[IoSlice::new(bytes)],
            &mut ancillary,
            SendFlags::NOSIGNAL,
        )
        .unwrap(),
        bytes.len()
    );
}

fn receive_challenge(peer: &OwnedFd) -> WorkerV3VerificationChallengeFrameV2 {
    let mut bytes = [0_u8; WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2];
    assert_eq!(
        recv(peer, &mut bytes, RecvFlags::empty()).unwrap().0,
        bytes.len()
    );
    WorkerV3VerificationChallengeFrameV2::decode_canonical(&bytes).unwrap()
}

fn expect_peer_eof(peer: &OwnedFd) {
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(1))
        .expect("bounded EOF deadline");
    let mut byte = [0_u8; 1];
    loop {
        match recv(peer, &mut byte, RecvFlags::empty()) {
            Ok((0, _)) => return,
            Ok((received, _)) => panic!("unexpected {received}-byte packet before EOF"),
            Err(Errno::AGAIN) if Instant::now() < deadline => thread::yield_now(),
            Err(error) => panic!("expected peer EOF: {error}"),
        }
    }
}

fn receive_terminal(peer: &OwnedFd) -> WorkerV3VerificationTerminalFrameV2 {
    let mut bytes = vec![0_u8; MAX_WORKER_V3_VERIFICATION_APPLICATION_RESPONSE_BYTES_V2 + 512];
    let count = recv(peer, &mut bytes, RecvFlags::empty()).unwrap().0;
    bytes.truncate(count);
    WorkerV3VerificationTerminalFrameV2::decode_canonical(&bytes).unwrap()
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
