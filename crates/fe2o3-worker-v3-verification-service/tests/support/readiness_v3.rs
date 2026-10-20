//! V3-only controls using the unchanged legacy fixture helpers.
use super::*;
use fe2o3_worker_v3_verification_client::{
    WorkerV3VerificationClientErrorV3, WorkerV3VerificationClientV3,
};
use fe2o3_worker_v3_verification_protocol::{
    WorkerV3VerificationProtocolErrorV3, WorkerV3VerificationReadyFrameV3,
};
use fe2o3_worker_v3_verification_service::{
    WorkerV3VerificationReadyServiceSessionV3, WorkerV3VerificationServiceErrorV3,
};

#[derive(Clone)]
struct CountedProviders(std::rc::Rc<std::cell::RefCell<[usize; 4]>>);
impl CountedProviders {
    fn new() -> Self {
        Self(std::rc::Rc::new(std::cell::RefCell::new([0; 4])))
    }
    fn assert_unused(&self) {
        assert_eq!(*self.0.borrow(), [0; 4]);
    }
}
impl WorkerV3VerificationPolicyResolverV1 for CountedProviders {
    fn resolve_expected_policy(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationPolicyIdentityV1> {
        self.0.borrow_mut()[0] += 1;
        Some(request.policy_identity())
    }
}
impl WorkerV3VerificationMeasurementResolverV1 for CountedProviders {
    fn resolve_expected_measurement(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        _: WorkerV3VerificationPolicyIdentityV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationMeasurementIdentityV1> {
        self.0.borrow_mut()[1] += 1;
        Some(request.measurement_identity())
    }
}
impl WorkerV3VerificationChallengeReplayGuardV1 for CountedProviders {
    fn admit_fresh_challenge(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        _: WorkerV3VerificationPolicyIdentityV1,
        _: WorkerV3VerificationFreshChallengeV1,
    ) -> bool {
        self.0.borrow_mut()[2] += 1;
        true
    }
}
impl WorkerV3VerificationChallengeReservationProviderV2 for CountedProviders {
    fn reserve_current_record_challenge(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        _: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationChallengeReservationV2> {
        self.0.borrow_mut()[3] += 1;
        Some(WorkerV3VerificationChallengeReservationV2::new([8; 32], [9; 32]).unwrap())
    }
}

fn connection() -> (tempfile::TempDir, PathBuf, OwnedFd, OwnedFd) {
    let (root, path, listener) = prepared_path_listener();
    let client = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
        None,
    )
    .unwrap();
    connect(&client, &SocketAddrUnix::new(&path).unwrap()).unwrap();
    rustix::net::sockopt::set_socket_passcred(&client, true).unwrap();
    let service = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    (root, path, service, client)
}
fn ready_bytes(client: &OwnedFd) -> [u8; 64] {
    let mut bytes = [0; 64];
    assert_eq!(
        recv(client, &mut bytes, RecvFlags::empty()).unwrap(),
        (64, 64)
    );
    bytes
}
fn send_composite(client: &OwnedFd, bytes: &[u8]) {
    let envelope = sealed_read_only(ENVELOPE);
    let hsaco = sealed_read_only(HSACO);
    let descriptors = [envelope.as_fd(), hsaco.as_fd()];
    let mut space =
        AlignedAncillaryStorage([MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2))]);
    let mut ancillary = SendAncillaryBuffer::new(&mut space.0);
    assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
    assert_eq!(
        sendmsg(
            client,
            &[IoSlice::new(bytes)],
            &mut ancillary,
            SendFlags::NOSIGNAL
        )
        .unwrap(),
        bytes.len()
    );
}
#[test]
fn readiness_precedes_begin_and_signed_current_record_preserves_deadline_and_output() {
    let request = verification_request(91);
    let fixture = CurrentRecordFixture::new();
    let (root, path, listener) = prepared_path_listener();
    let deadline = Instant::now() + Duration::from_secs(5);
    let request_for_client = request.clone();
    let path_for_client = path.clone();
    let client = thread::spawn(move || {
        let fd = socket_with(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
            None,
        )
        .unwrap();
        connect(&fd, &SocketAddrUnix::new(&path_for_client).unwrap()).unwrap();
        let client = WorkerV3VerificationClientV3::admit_connected_path_until(
            fd,
            &path_for_client,
            deadline,
        )
        .unwrap();
        assert_eq!(client.deadline(), deadline);
        assert!(!client.grants_authority());
        let ready = client.wait_ready().unwrap();
        assert_eq!(ready.deadline(), deadline);
        assert!(!ready.grants_authority());
        let ClientBeginOutcomeV2::Reserved(reserved) = ready
            .begin(request_for_client.clone(), snapshots(&request_for_client))
            .unwrap()
        else {
            panic!("V3 Begin refused")
        };
        let (challenge, pending) = reserved.into_parts();
        assert_eq!(pending.deadline(), deadline);
        let (verification, attestation) = fixture.records(challenge.into_bytes());
        pending
            .submit_current_record(
                *verification.canonical_bytes(),
                *attestation.canonical_bytes(),
            )
            .unwrap()
    });
    let accepted = accept_with(&listener, SocketFlags::CLOEXEC | SocketFlags::NONBLOCK).unwrap();
    let endpoint = WorkerV3VerificationAcceptedServiceEndpointV2::admit(accepted, &path).unwrap();
    let ready =
        WorkerV3VerificationReadyServiceSessionV3::prepare_until(endpoint, deadline).unwrap();
    assert_eq!(ready.deadline(), deadline);
    assert!(!ready.grants_authority());
    let (mut policy, mut measurement, mut replay) = admission_state(&request);
    let mut reservations = FixedReservations::available(0x81, 0x82);
    let WorkerV3VerificationBeginOutcomeV2::Reserved(pending) = ready
        .begin(
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations,
        )
        .unwrap()
    else {
        panic!("V3 Begin refused")
    };
    assert_eq!(pending.deadline(), deadline);
    let WorkerV3VerificationCurrentRecordOutcomeV2::Ready(terminal) =
        pending.receive_current_record().unwrap()
    else {
        panic!("signed current record refused")
    };
    assert_eq!(terminal.deadline(), deadline);
    assert_eq!(
        terminal
            .payload(WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2)
            .sha256(),
        &sha256(ENVELOPE)
    );
    assert!(!terminal.grants_authority());
    terminal
        .send_application_response(b"v3-signed-current-output".to_vec())
        .unwrap();
    let output = client.join().unwrap();
    assert_eq!(
        output.disposition(),
        WorkerV3VerificationTerminalDispositionV2::ApplicationResponse
    );
    assert_eq!(
        output.application_response_bytes(),
        b"v3-signed-current-output"
    );
    assert!(!output.grants_authority());
    drop(root);
}
#[test]
fn send_first_is_rejected_by_second_strict_check_and_retains_cleanup_fd() {
    let (_root, path, service, client) = connection();
    let endpoint = WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap();
    send(&client, b"early", SendFlags::NOSIGNAL).unwrap();
    let failure = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
        endpoint,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap_err();
    assert!(!failure.ready_send_attempted());
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV3::V1(WorkerV3VerificationServiceErrorV1::InvalidControl(
            _
        ))
    ));
    let retained = failure.into_control().expect("pre-I/O cleanup fd retained");
    expect_peer_eof(&client);
    assert!(send(&client, b"retry", SendFlags::NOSIGNAL).is_err());
    assert!(send(&retained, b"retry", SendFlags::NOSIGNAL).is_err());
    drop(retained);
}
#[test]
fn wrong_stale_nonce_and_legacy_begin_fail_before_policy_or_reservation() {
    for mode in 0..3 {
        let (_root, path, service, client) = connection();
        let endpoint =
            WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap();
        let ready = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
            endpoint,
            Instant::now() + Duration::from_secs(2),
        )
        .unwrap();
        let bytes = ready_bytes(&client);
        let frame = WorkerV3VerificationReadyFrameV3::decode_canonical(&bytes).unwrap();
        let request = verification_request(92);
        let mut packet = frame.encode_begin(&request);
        if mode == 0 {
            packet[16] ^= 1;
        }
        if mode == 1 {
            packet[8] = 2;
        }
        if mode == 2 {
            packet = request.encode_canonical().to_vec();
        }
        send_composite(&client, &packet);
        let counted = CountedProviders::new();
        let (mut policy, mut measurement, mut replay, mut reservations) = (
            counted.clone(),
            counted.clone(),
            counted.clone(),
            counted.clone(),
        );
        let err = ready
            .begin(
                &mut policy,
                &mut measurement,
                &mut replay,
                &mut reservations,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            WorkerV3VerificationServiceErrorV3::Protocol(_)
        ));
        counted.assert_unused();
        expect_peer_eof(&client);
    }
}
#[test]
fn cross_connection_nonce_is_not_a_begin_capability() {
    let (_a, path_a, service_a, client_a) = connection();
    let ready_a = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
        WorkerV3VerificationAcceptedServiceEndpointV2::admit(service_a, &path_a).unwrap(),
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    let old_frame =
        WorkerV3VerificationReadyFrameV3::decode_canonical(&ready_bytes(&client_a)).unwrap();
    let (_b, path_b, service_b, client_b) = connection();
    let ready_b = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
        WorkerV3VerificationAcceptedServiceEndpointV2::admit(service_b, &path_b).unwrap(),
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    let fresh_frame =
        WorkerV3VerificationReadyFrameV3::decode_canonical(&ready_bytes(&client_b)).unwrap();
    assert_ne!(old_frame, fresh_frame);
    let request = verification_request(93);
    send_composite(&client_b, &old_frame.encode_begin(&request));
    let counted = CountedProviders::new();
    let (mut policy, mut measurement, mut replay, mut reservations) = (
        counted.clone(),
        counted.clone(),
        counted.clone(),
        counted.clone(),
    );
    assert!(matches!(
        ready_b.begin(
            &mut policy,
            &mut measurement,
            &mut replay,
            &mut reservations
        ),
        Err(WorkerV3VerificationServiceErrorV3::Protocol(
            WorkerV3VerificationProtocolErrorV3::NonceMismatch
        ))
    ));
    expect_peer_eof(&client_b);
    counted.assert_unused();
    drop(ready_a);
}
#[test]
fn preexpired_prepare_retains_fd_and_emits_no_ready() {
    let (_root, path, service, client) = connection();
    let endpoint = WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap();
    let failure =
        WorkerV3VerificationReadyServiceSessionV3::prepare_until(endpoint, Instant::now())
            .unwrap_err();
    assert!(!failure.ready_send_attempted());
    let fd = failure.into_control().unwrap();
    expect_peer_eof(&client);
    assert!(send(&client, b"retry", SendFlags::NOSIGNAL).is_err());
    assert!(send(&fd, b"retry", SendFlags::NOSIGNAL).is_err());
    drop(fd);
}

#[test]
fn duplicate_begin_is_rejected_before_any_provider_effects() {
    let (_root, path, service, client) = connection();
    let ready = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
        WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap(),
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    let frame = WorkerV3VerificationReadyFrameV3::decode_canonical(&ready_bytes(&client)).unwrap();
    let packet = frame.encode_begin(&verification_request(95));
    send_composite(&client, &packet);
    send_composite(&client, &packet);
    let counted = CountedProviders::new();
    let WorkerV3VerificationBeginOutcomeV2::Rejected(rejected) = ready
        .begin(
            &mut counted.clone(),
            &mut counted.clone(),
            &mut counted.clone(),
            &mut counted.clone(),
        )
        .unwrap()
    else {
        panic!("duplicate Begin reserved a challenge");
    };
    assert_eq!(
        rejected.reason(),
        WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder
    );
    counted.assert_unused();
    let expected = WorkerV3VerificationChallengeFrameV2::rejected(&verification_request(95));
    assert_eq!(
        rejected.frame().encode_canonical(),
        expected.encode_canonical()
    );
    // Unread duplicate payload may reset the connection before rejection delivery.
    // The extra byte ensures an oversized packet cannot masquerade as a canonical prefix.
    let mut bytes = [0; WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2 + 1];
    match recv(&client, &mut bytes, RecvFlags::DONTWAIT) {
        Err(Errno::CONNRESET) => {}
        Ok((received, _)) => {
            assert_eq!(received, WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2);
            let frame =
                WorkerV3VerificationChallengeFrameV2::decode_canonical(&bytes[..received]).unwrap();
            assert_eq!(frame.disposition(), fe2o3_worker_v3_verification_protocol::WorkerV3VerificationChallengeDispositionV2::Rejected);
            assert_eq!(frame.encode_canonical(), expected.encode_canonical());
            let mut byte = [0];
            assert!(matches!(
                recv(&client, &mut byte, RecvFlags::DONTWAIT),
                Ok((0, 0)) | Err(Errno::CONNRESET)
            ));
        }
        other => panic!("unexpected duplicate-Begin terminal receive: {other:?}"),
    }
}
#[test]
fn client_rejects_malformed_legacy_ready_and_extra_fds() {
    for mode in 0..4 {
        let (_root, path, service, client) = connection();
        let client = WorkerV3VerificationClientV3::admit_connected_path_until(
            client,
            &path,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        let mut bytes = WorkerV3VerificationReadyFrameV3::new([4; 32])
            .unwrap()
            .encode_canonical()
            .to_vec();
        if mode == 0 {
            bytes[0] ^= 1;
        }
        if mode == 1 {
            bytes[8] = 2;
        }
        if mode == 2 {
            bytes.push(0);
        }
        if mode == 3 {
            send_composite(&service, &bytes);
        } else {
            send(&service, &bytes, SendFlags::NOSIGNAL).unwrap();
        }
        let err = client.wait_ready().unwrap_err();
        assert!(matches!(
            err,
            WorkerV3VerificationClientErrorV3::Protocol(_)
                | WorkerV3VerificationClientErrorV3::V2(_)
        ));
    }
}
#[test]
fn queued_ready_is_credential_checked_after_client_enables_receiving() {
    let (_root, path, service, client) = connection();
    rustix::net::sockopt::set_socket_passcred(&client, false).unwrap();
    // Simulate server scheduling first; it already has mandatory inherited SO_PASSCRED.
    let endpoint = WorkerV3VerificationAcceptedServiceEndpointV2::admit(service, &path).unwrap();
    let ready = WorkerV3VerificationReadyServiceSessionV3::prepare_until(
        endpoint,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    let client = WorkerV3VerificationClientV3::admit_connected_path_until(
        client,
        &path,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    assert!(!client.wait_ready().unwrap().grants_authority());
    drop(ready);
}

#[test]
fn duplicate_ready_and_eof_queued_before_begin_never_send_begin() {
    for mode in 0..3 {
        let (_root, path, service, client) = connection();
        let client = WorkerV3VerificationClientV3::admit_connected_path_until(
            client,
            &path,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        let bytes = WorkerV3VerificationReadyFrameV3::new([5; 32])
            .unwrap()
            .encode_canonical();
        send(&service, &bytes, SendFlags::NOSIGNAL).unwrap();
        if mode == 0 {
            send(&service, &bytes, SendFlags::NOSIGNAL).unwrap();
            assert!(client.wait_ready().is_err());
        } else {
            let ready = client.wait_ready().unwrap();
            if mode == 1 {
                send(&service, &bytes, SendFlags::NOSIGNAL).unwrap();
            } else {
                shutdown(&service, Shutdown::Write).unwrap();
            }
            let request = verification_request(94);
            assert!(ready.begin(request.clone(), snapshots(&request)).is_err());
        }
        let mut byte = [0u8; 1];
        match recv(&service, &mut byte, RecvFlags::DONTWAIT) {
            Ok((0, 0)) | Err(Errno::CONNRESET) => {}
            other => panic!("client sent data before rejecting queued readiness: {other:?}"),
        }
    }
}
#[test]
fn rejected_ready_closes_even_truncated_ancillary_rights() {
    use std::io::Read;
    for count in [1, 3] {
        let (_root, path, service, client) = connection();
        let client = WorkerV3VerificationClientV3::admit_connected_path_until(
            client,
            &path,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        let (a, mut witness) = std::os::unix::net::UnixStream::pair().unwrap();
        witness
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let descriptors = vec![a.as_fd(); count];
        let mut space =
            AlignedAncillaryStorage([MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(3))]);
        let mut ancillary = SendAncillaryBuffer::new(&mut space.0);
        assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
        let bytes = WorkerV3VerificationReadyFrameV3::new([6; 32])
            .unwrap()
            .encode_canonical();
        sendmsg(
            &service,
            &[IoSlice::new(&bytes)],
            &mut ancillary,
            SendFlags::NOSIGNAL,
        )
        .unwrap();
        drop(descriptors);
        drop(a);
        assert!(client.wait_ready().is_err());
        let mut byte = [0];
        assert_eq!(
            witness.read(&mut byte).unwrap(),
            0,
            "a received ancillary fd leaked"
        );
    }
}
#[test]
fn ready_sent_by_post_connect_descriptor_recipient_has_wrong_credentials() {
    let (_root, path, service, client) = connection();
    let client = WorkerV3VerificationClientV3::admit_connected_path_until(
        client,
        &path,
        Instant::now() + Duration::from_secs(3),
    )
    .unwrap();
    let bytes = WorkerV3VerificationReadyFrameV3::new([7; 32])
        .unwrap()
        .encode_canonical();
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let mut child = Command::new("/usr/bin/python3")
        .arg("-I")
        .arg("-c")
        .arg("import socket,sys; s=socket.socket(fileno=0); s.send(bytes.fromhex(sys.argv[1]))")
        .arg(hex)
        .stdin(Stdio::from(File::from(service)))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let err = client.wait_ready().unwrap_err();
    assert!(matches!(
        err,
        WorkerV3VerificationClientErrorV3::V2(
            WorkerV3VerificationClientErrorV2::ResponseCredentialsMismatch
        )
    ));
    assert!(child.wait().unwrap().success());
}
