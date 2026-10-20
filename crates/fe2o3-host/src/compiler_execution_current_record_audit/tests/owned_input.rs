use super::*;
use std::cell::Cell;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::time::Instant;

type Auditor = InheritedWorkerV3CompilerCurrentRecordAuditorV1;

fn identity(peer: &OwnedFd) -> (u64, u64) {
    let metadata = rustix::fs::fstat(peer).unwrap();
    (metadata.st_dev, metadata.st_ino)
}

fn matching_slots((device, inode): (u64, u64)) -> Vec<i32> {
    fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let metadata = fs::metadata(entry.path()).ok()?;
            if metadata.dev() == device && metadata.ino() == inode {
                entry.file_name().to_str()?.parse().ok()
            } else {
                None
            }
        })
        .collect()
}

fn descriptor_flags(raw: i32) -> std::io::Result<i32> {
    // F_GETFD only queries scalar metadata; it neither borrows nor closes this slot.
    let flags = unsafe { libc::fcntl(raw, libc::F_GETFD) };
    if flags < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(flags)
    }
}

fn assert_retired(raw: i32) {
    assert_eq!(
        descriptor_flags(raw).unwrap_err().raw_os_error(),
        Some(libc::EBADF)
    );
}

#[test]
fn owned_admission_retires_input_and_retains_only_a_cloexec_duplicate() {
    let (peer, service) = socket_pair();
    let input = peer.as_raw_fd();
    let object = identity(&peer);
    let auditor = Auditor::admit_owned_application_service(peer).unwrap();
    assert_retired(input);
    let slots = matching_slots(object);
    assert_eq!(slots.len(), 1);
    assert_ne!(slots[0], input);
    assert_ne!(descriptor_flags(slots[0]).unwrap() & libc::FD_CLOEXEC, 0);

    let replacement = rustix::io::fcntl_dupfd_cloexec(&service, input).unwrap();
    assert_eq!(replacement.as_raw_fd(), input);
    drop(auditor);
    assert!(matching_slots(object).is_empty());
    assert!(rustix::fs::fstat(&replacement).is_ok());
}

#[test]
fn invalid_owned_peers_close_input_and_retained_duplicate() {
    use std::os::unix::net::{UnixDatagram, UnixStream};

    let (stream, _stream_service) = UnixStream::pair().unwrap();
    let (datagram, _datagram_service) = UnixDatagram::pair().unwrap();
    let unconnected = rustix::net::socket_with(
        rustix::net::AddressFamily::UNIX,
        rustix::net::SocketType::SEQPACKET,
        rustix::net::SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    let peers: Vec<OwnedFd> = vec![
        fs::File::open("/dev/null").unwrap().into(),
        stream.into(),
        datagram.into(),
        unconnected,
    ];
    for peer in peers {
        let input = peer.as_raw_fd();
        let object = identity(&peer);
        let before = matching_slots(object).len();
        assert!(Auditor::admit_owned_application_service(peer).is_err());
        assert_retired(input);
        assert_eq!(matching_slots(object).len(), before - 1);
    }
}

#[test]
fn duplicate_failure_consumes_input_in_an_isolated_child() {
    const CHILD: &str = "FE2O3_OWNED_AUDIT_DUP_LIMIT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};

        let (peer, service) = socket_pair();
        let input = peer.as_raw_fd();
        let object = identity(&peer);
        let service_object = identity(&service);
        let service_slots = matching_slots(service_object).len();
        let original = getrlimit(Resource::Nofile);
        setrlimit(
            Resource::Nofile,
            Rlimit {
                current: Some(64),
                maximum: original.maximum,
            },
        )
        .unwrap();
        // A valid minimum plus occupied eligible slots forces EMFILE, not EINVAL.
        let mut fillers = Vec::new();
        let mut exhaustion = None;
        for _ in 0..64 {
            match rustix::io::fcntl_dupfd_cloexec(&service, 3) {
                Ok(filler) => fillers.push(filler),
                Err(error) => {
                    exhaustion = std::io::Error::from(error).raw_os_error();
                    break;
                }
            }
        }
        if exhaustion != Some(libc::EMFILE) {
            setrlimit(Resource::Nofile, original).unwrap();
            panic!("expected genuine descriptor exhaustion, got {exhaustion:?}");
        }
        let result = Auditor::admit_owned_application_service(peer);
        setrlimit(Resource::Nofile, original).unwrap();
        assert!(matches!(
            result,
            Err(CompilerExecutionClientErrorV1::Descriptor(error))
                if error.raw_os_error() == Some(libc::EMFILE)
        ));
        assert_retired(input);
        assert!(matching_slots(object).is_empty());
        for filler in &fillers {
            assert!((3..64).contains(&filler.as_raw_fd()));
            assert_ne!(
                descriptor_flags(filler.as_raw_fd()).unwrap() & libc::FD_CLOEXEC,
                0
            );
        }
        drop(fillers);
        assert_eq!(matching_slots(service_object).len(), service_slots);
        return;
    }

    let qualified = concat!(
        module_path!(),
        "::duplicate_failure_consumes_input_in_an_isolated_child"
    );
    let (_, name) = qualified.split_once("::").unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--test-threads=1", "--nocapture"])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn unwinding_after_admission_releases_retained_custody() {
    let (peer, _service) = socket_pair();
    let input = peer.as_raw_fd();
    let object = identity(&peer);
    let retained = Cell::new(-1);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _auditor = Auditor::admit_owned_application_service(peer).unwrap();
        retained.set(matching_slots(object)[0]);
        panic!("synthetic owned-auditor unwind");
    }));
    assert!(result.is_err());
    assert_retired(input);
    assert_retired(retained.get());
    assert!(matching_slots(object).is_empty());
}

#[test]
fn failed_request_consumes_session_and_releases_peer() {
    let fixture = Fixture::new(0x20);
    let (peer, _service) = socket_pair();
    let object = identity(&peer);
    let mut auditor = Auditor::admit_owned_application_service(peer).unwrap();
    assert!(matches!(
        auditor.audit_exact(&subject(0x21), &fixture.carriage),
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::RequestMismatch)
    ));
    assert!(matching_slots(object).is_empty());
    assert!(matches!(
        auditor.audit_exact(&fixture.subject, &fixture.carriage),
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::AlreadyConsumed)
    ));
}

#[test]
fn safe_socket_alias_interference_fails_closed_without_closing_the_alias() {
    let fixture = Fixture::new(0x20);
    let (peer, _service) = socket_pair();
    let alias = peer.try_clone().unwrap();
    let mut auditor = Auditor::admit_owned_application_service(peer).unwrap();
    rustix::net::shutdown(&alias, rustix::net::Shutdown::Both).unwrap();
    assert!(matches!(
        auditor.audit_exact(&fixture.subject, &fixture.carriage),
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::Client(_))
    ));
    assert!(matches!(
        auditor.audit_exact(&fixture.subject, &fixture.carriage),
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::AlreadyConsumed)
    ));
    assert!(rustix::fs::fstat(&alias).is_ok());
}

#[test]
fn signed_response_with_another_challenge_is_rejected_and_consumed() {
    let fixture = Fixture::new(0x20);
    let (peer, service) = socket_pair();
    let mut auditor = Auditor::admit_owned_application_service(peer).unwrap();
    let worker = thread::spawn(move || {
        let request = receive_request(&service);
        let service_fixture = Fixture::new(0x20);
        let attestation = direct_current_record_attestation(&service_fixture, [0xe1; 32]);
        let response =
            CompilerExecutionServiceResponseV1::verified_current(request.identity(), attestation)
                .unwrap();
        send_response(&service, response.canonical_bytes());
    });
    let result = auditor.audit_exact_with_challenge(
        &fixture.subject,
        &fixture.carriage,
        CompilerExecutionCurrentRecordChallengeV1::from_bytes([0xe2; 32]).unwrap(),
    );
    worker.join().unwrap();
    assert!(matches!(
        result,
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::Client(_))
    ));
    assert!(matches!(
        auditor.audit_exact(&fixture.subject, &fixture.carriage),
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::AlreadyConsumed)
    ));
}

#[test]
fn owned_session_has_the_fixed_thirty_second_deadline() {
    let _: fn(OwnedFd) -> Result<Auditor, CompilerExecutionClientErrorV1> =
        Auditor::admit_owned_application_service;
    assert_eq!(
        WORKER_V3_COMPILER_CURRENT_RECORD_AUDIT_TIMEOUT_V1,
        Duration::from_secs(30)
    );
    let fixture = Fixture::new(0x20);
    let (peer, _silent_service) = socket_pair();
    let mut auditor = Auditor::admit_owned_application_service(peer).unwrap();
    let start = Instant::now();
    let result = auditor.audit_exact(&fixture.subject, &fixture.carriage);
    let elapsed = start.elapsed();
    assert!(matches!(
        result,
        Err(WorkerV3CompilerCurrentRecordAuditErrorV1::Client(
            CompilerExecutionClientErrorV1::Timeout
        ))
    ));
    assert!(elapsed >= Duration::from_secs(29), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(35), "{elapsed:?}");
}
