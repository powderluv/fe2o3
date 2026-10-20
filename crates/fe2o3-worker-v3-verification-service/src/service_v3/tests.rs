use super::*;
use fe2o3_worker_v3_verification_protocol::*;
use rustix::fs::{MemfdFlags, Mode, OFlags, SealFlags};
use rustix::net::{self, AddressFamily, RecvFlags, SendFlags, Shutdown, SocketFlags, SocketType};
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::Write,
    os::fd::AsRawFd,
    rc::Rc,
    time::Duration,
};

const ENVELOPE: &[u8] = b"envelope";
const HSACO: &[u8] = b"hsaco";

fn pair() -> (OwnedFd, OwnedFd, WorkerV3VerificationCallerV1) {
    let (service, peer) = net::socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    net::sockopt::set_socket_passcred(&service, true).unwrap();
    let caller = crate::service::caller_identity(&service).unwrap();
    (service, peer, caller)
}
fn eof(peer: &OwnedFd) {
    let mut byte = [0];
    assert_eq!(
        net::recv(peer, &mut byte, RecvFlags::DONTWAIT).unwrap(),
        (0, 0)
    );
}
fn request() -> WorkerV3VerificationRequestV1 {
    WorkerV3VerificationRequestV1::new(
        WorkerV3VerificationFreshChallengeV1::new([1; 32]).unwrap(),
        WorkerV3VerificationRosterIdentityV1::new([2; 32]).unwrap(),
        WorkerV3VerificationPolicyIdentityV1::new([3; 32]).unwrap(),
        WorkerV3VerificationMeasurementIdentityV1::new([4; 32]).unwrap(),
        WorkerV3VerificationFdPayloadDescriptorV1::load_envelope_v2(
            ENVELOPE.len() as u64,
            Sha256::digest(ENVELOPE).into(),
        )
        .unwrap(),
        WorkerV3VerificationFdPayloadDescriptorV1::finalized_hsaco(
            HSACO.len() as u64,
            Sha256::digest(HSACO).into(),
        )
        .unwrap(),
        vec![
            WorkerV3VerificationEntryCoordinateV1::new(0, "a", "b", [1; 32], [2; 32], [3; 32])
                .unwrap(),
        ],
    )
    .unwrap()
}
fn sealed(bytes: &[u8]) -> OwnedFd {
    let mut writer = File::from(
        rustix::fs::memfd_create(
            "readiness4524",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .unwrap(),
    );
    rustix::fs::fchmod(&writer, Mode::RUSR).unwrap();
    writer.write_all(bytes).unwrap();
    rustix::fs::fcntl_add_seals(
        &writer,
        SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL,
    )
    .unwrap();
    rustix::fs::open(
        format!("/proc/self/fd/{}", writer.as_raw_fd()),
        OFlags::RDONLY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .unwrap()
}
#[derive(Clone)]
struct Effects(Rc<RefCell<[usize; 4]>>);
impl WorkerV3VerificationPolicyResolverV1 for Effects {
    fn resolve_expected_policy(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationPolicyIdentityV1> {
        self.0.borrow_mut()[0] += 1;
        Some(request.policy_identity())
    }
}
impl WorkerV3VerificationMeasurementResolverV1 for Effects {
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
impl WorkerV3VerificationChallengeReplayGuardV1 for Effects {
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
impl WorkerV3VerificationChallengeReservationProviderV2 for Effects {
    fn reserve_current_record_challenge(
        &mut self,
        _: WorkerV3VerificationCallerV1,
        _: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationChallengeReservationV2> {
        self.0.borrow_mut()[3] += 1;
        Some(WorkerV3VerificationChallengeReservationV2::new([8; 32], [9; 32]).unwrap())
    }
}

#[test]
fn all_three_shared_probes_expire_iteratively_without_later_effects_or_deadline_reset() {
    for failing_site in 0..3 {
        let (control, peer, caller) = pair();
        let deadline = Instant::now() + Duration::from_secs(60);
        let calls = Cell::new(0);
        let interrupts = Cell::new(0);
        let effects = Effects(Rc::new(RefCell::new([0; 4])));
        let result = begin_received_worker_v3_verification_session_until_v2(
            (control, caller, deadline, |_: &OwnedFd, observed| {
                assert_eq!(observed, deadline, "original absolute deadline changed");
                let site = calls.get();
                calls.set(site + 1);
                if site != failing_site {
                    return Ok(false);
                }
                phase_probe_with(
                    observed,
                    || {
                        if interrupts.get() < 4096 {
                            observed - Duration::from_secs(1)
                        } else {
                            observed
                        }
                    },
                    || {
                        interrupts.set(interrupts.get() + 1);
                        Err(rustix::io::Errno::INTR)
                    },
                )
            }),
            request(),
            [sealed(ENVELOPE), sealed(HSACO)],
            &mut effects.clone(),
            &mut effects.clone(),
            &mut effects.clone(),
            &mut effects.clone(),
        );
        assert!(matches!(
            result,
            Err(WorkerV3VerificationServiceErrorV2::V1(
                WorkerV3VerificationServiceErrorV1::Timeout
            ))
        ));
        assert_eq!(calls.get(), failing_site + 1);
        assert_eq!(interrupts.get(), 4096);
        assert_eq!(
            *effects.0.borrow(),
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]][failing_site]
        );
        eof(&peer);
    }
}

#[test]
fn phase_probe_checks_expiry_even_after_a_successful_poll() {
    let deadline = Instant::now() + Duration::from_secs(60);
    let polled = Cell::new(false);
    let result = phase_probe_with(
        deadline,
        || {
            if polled.get() {
                deadline
            } else {
                deadline - Duration::from_secs(1)
            }
        },
        || {
            polled.set(true);
            Ok(false)
        },
    );
    assert!(matches!(
        result,
        Err(WorkerV3VerificationServiceErrorV2::V1(
            WorkerV3VerificationServiceErrorV1::Timeout
        ))
    ));
}
#[test]
fn phase_probe_recovers_from_finite_interruptions_and_preserves_io_errors() {
    let deadline = Instant::now() + Duration::from_secs(60);
    let count = Cell::new(0);
    assert!(
        !phase_probe_with(deadline, Instant::now, || {
            count.set(count.get() + 1);
            if count.get() < 1024 {
                Err(rustix::io::Errno::INTR)
            } else {
                Ok(false)
            }
        })
        .unwrap()
    );
    assert_eq!(count.get(), 1024);
    assert!(matches!(
        phase_probe_with(deadline, Instant::now, || Err(rustix::io::Errno::BADF)),
        Err(WorkerV3VerificationServiceErrorV2::Io { .. })
    ));
}
#[test]
fn actual_early_input_branches_shutdown_retained_cleanup_and_aliases() {
    // This private entry is reached only after both public strict endpoint checks.
    // Inject at its two real queue probes, not at a fixture readiness barrier.
    for during_entropy in [false, true] {
        let (control, peer, caller) = pair();
        let alias = rustix::io::dup(&control).unwrap();
        if !during_entropy {
            net::send(&peer, b"early", SendFlags::NOSIGNAL).unwrap();
        }
        let failure = WorkerV3VerificationReadyServiceSessionV3::prepare_checked_until(
            control,
            caller,
            Instant::now() + Duration::from_secs(60),
            |nonce| {
                assert!(during_entropy, "entropy ran after early-input refusal");
                nonce.fill(1);
                net::send(&peer, b"early", SendFlags::NOSIGNAL).unwrap();
                Ok(32)
            },
        )
        .unwrap_err();
        assert!(matches!(
            failure.source_error(),
            WorkerV3VerificationServiceErrorV3::EarlyInput
        ));
        assert!(!failure.ready_send_attempted());
        let retained = failure.into_control().unwrap();
        eof(&peer);
        assert!(net::send(&peer, b"retry", SendFlags::NOSIGNAL).is_err());
        assert!(net::send(&alias, b"retry", SendFlags::NOSIGNAL).is_err());
        drop((retained, alias));
    }
}
#[test]
fn attempted_ready_send_failure_shuts_down_retained_fd_and_alias() {
    let (control, peer, caller) = pair();
    let alias = rustix::io::dup(&control).unwrap();
    let failure = WorkerV3VerificationReadyServiceSessionV3::prepare_checked_until(
        control,
        caller,
        Instant::now() + Duration::from_secs(60),
        |nonce| {
            nonce.fill(1);
            net::shutdown(&peer, Shutdown::Read).unwrap();
            Ok(32)
        },
    )
    .unwrap_err();
    assert!(failure.ready_send_attempted());
    assert!(matches!(
        failure.source_error(),
        WorkerV3VerificationServiceErrorV3::V1(_)
    ));
    let retained = failure.into_control().unwrap();
    eof(&peer);
    // Peer read shutdown alone does not cause this: service cleanup must shut its read half.
    assert!(net::send(&peer, b"retry", SendFlags::NOSIGNAL).is_err());
    assert!(net::send(&alias, b"retry", SendFlags::NOSIGNAL).is_err());
    drop((retained, alias));
}
#[test]
fn expiry_after_emitted_ready_refuses_begin_and_never_calls_providers() {
    let (control, peer, caller) = pair();
    let ready = WorkerV3VerificationReadyServiceSessionV3::prepare_checked_until(
        control,
        caller,
        Instant::now() + Duration::from_secs(60),
        |nonce| {
            nonce.fill(1);
            Ok(32)
        },
    )
    .unwrap();
    let mut bytes = [0; 64];
    assert_eq!(
        net::recv(&peer, &mut bytes, RecvFlags::DONTWAIT).unwrap(),
        (64, 64)
    );
    WorkerV3VerificationReadyFrameV3::decode_canonical(&bytes).unwrap();
    // Deterministically model the already elapsed absolute Instant without sleeping.
    let expired = WorkerV3VerificationReadyServiceSessionV3 {
        deadline: Instant::now(),
        ..ready
    };
    let effects = Effects(Rc::new(RefCell::new([0; 4])));
    assert!(matches!(
        expired.begin(
            &mut effects.clone(),
            &mut effects.clone(),
            &mut effects.clone(),
            &mut effects.clone()
        ),
        Err(WorkerV3VerificationServiceErrorV3::V2(
            WorkerV3VerificationServiceErrorV2::V1(WorkerV3VerificationServiceErrorV1::Timeout)
        ))
    ));
    assert_eq!(*effects.0.borrow(), [0; 4]);
    eof(&peer);
}
