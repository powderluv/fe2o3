//! Admission refusals need no socket fixture and run before transport I/O.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use rustix::pipe::{PipeFlags, pipe_with};

pub(super) fn peer() -> (OwnedFd, OwnedFd) {
    pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap()
}

pub(super) fn closed(reader: OwnedFd) {
    wait_for_pipe_eof(&reader, Duration::from_secs(2), || {}).unwrap();
}

fn wait_for_pipe_eof(
    reader: &OwnedFd,
    timeout: Duration,
    mut pending: impl FnMut(),
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| Error::other("EOF deadline overflow"))?;
    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorKind::TimedOut, "pipe EOF not observed"));
        }
        match rustix::io::read(reader, &mut [0]) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(Error::new(ErrorKind::InvalidData, "data before pipe EOF")),
            Err(error) => {
                let error = Error::from(error);
                if !matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) {
                    return Err(error);
                }
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(Error::new(ErrorKind::TimedOut, "pipe EOF not observed"));
        }
        // Concurrent fork/exec can briefly inherit a CLOEXEC writer. Only a
        // zero-byte read proves EOF; pending readiness or a deadline never does.
        pending();
        std::thread::sleep(remaining.min(Duration::from_millis(1)));
    }
}

#[test]
fn pipe_eof_waits_for_actual_close_of_a_delayed_duplicate() {
    let (reader, writer) = peer();
    let mut duplicate = Some(rustix::io::fcntl_dupfd_cloexec(&writer, 0).unwrap());
    drop(writer);
    let mut observed_pending = false;
    wait_for_pipe_eof(&reader, Duration::from_secs(2), || {
        observed_pending = true;
        drop(duplicate.take());
    })
    .unwrap();
    assert!(observed_pending);
    assert!(duplicate.is_none());
}

#[test]
fn pipe_eof_refuses_a_retained_writer_at_the_deadline() {
    let (reader, writer) = peer();
    let error = wait_for_pipe_eof(&reader, Duration::from_millis(10), || {}).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert_eq!(rustix::io::write(&writer, b"x").unwrap(), 1);
}

#[test]
fn pipe_eof_refuses_data_even_after_the_writer_closes() {
    let (reader, writer) = peer();
    assert_eq!(rustix::io::write(&writer, b"x").unwrap(), 1);
    drop(writer);
    let error = wait_for_pipe_eof(&reader, Duration::from_secs(2), || {}).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn admission_resource_denials_close_input_and_keep_original_account() {
    for (limit, prepaid) in [
        (7, TestClient::PEER_STORAGE),
        (64 * 1024, TestClient::PEER_STORAGE - 1),
    ] {
        let (reader, peer) = peer();
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, 64 * 1024);
        budget.reserve_storage(prepaid).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = TestClient::admit(peer, Duration::from_secs(1), &mut budget);
        if limit == 7 {
            assert!(matches!(
                result,
                Err(ClientError::Resource(Resource::Work(_)))
            ));
        } else {
            assert!(matches!(
                result,
                Err(ClientError::Resource(Resource::Accounting))
            ));
        }
        drop(result);
        assert_eq!(budget.storage(), prepaid);
        assert!(budget.work_ledger_identity_v1() == ledger);
        closed(reader);
    }
}

#[test]
fn invalid_timeout_is_terminal_and_does_not_refund_work() {
    let mut work = Work::new(1_000_000);
    let mut budget = Budget::new(&mut work, 64 * 1024);
    budget.reserve_storage(TestClient::PEER_STORAGE).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    for timeout in [Duration::ZERO, Duration::from_secs(301)] {
        let (reader, peer) = peer();
        let used = budget.work();
        assert!(matches!(
            TestClient::admit(peer, timeout, &mut budget),
            Err(ClientError::Transport(TransportError::InvalidTimeout))
        ));
        assert_eq!(budget.storage(), TestClient::PEER_STORAGE);
        assert_eq!(budget.work(), used + 64 * 1024);
        assert!(budget.work_ledger_identity_v1() == ledger);
        closed(reader);
    }
}
