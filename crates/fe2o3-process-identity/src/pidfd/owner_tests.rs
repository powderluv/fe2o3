use super::*;
use std::os::fd::FromRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

fn pidfd_for(pid: u32) -> OwnedFd {
    // SAFETY: scalar arguments; success returns an exclusively owned descriptor.
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    assert!(raw >= 0, "pidfd_open: {}", io::Error::last_os_error());
    // SAFETY: the successful syscall returned a new, owned descriptor.
    unsafe { OwnedFd::from_raw_fd(raw as i32) }
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn sleeping_child() -> ChildGuard {
    ChildGuard(Command::new("/bin/sleep").arg("30").spawn().unwrap())
}

fn await_exit(fd: &OwnedFd) {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        assert!(Instant::now() < deadline, "process exit timed out");
        // SAFETY: one initialized pollfd, kept borrowed throughout the call.
        let result = unsafe { libc::poll(&mut poll, 1, 10) };
        assert!(result >= 0, "poll: {}", io::Error::last_os_error());
        if result == 1 {
            break;
        }
    }
}

#[test]
fn received_owner_checks_original_process_and_rejects_wrong_pid_and_non_pidfd() {
    let pid = std::process::id();
    let owner = ReceivedProcessPidfdV1::admit_received(pidfd_for(pid), pid).unwrap();
    owner.revalidate().unwrap();
    assert_eq!(owner.pid(), pid);
    assert_eq!(
        owner.start_time_ticks(),
        current_process_start_time_ticks_v1().unwrap()
    );
    assert!(!format!("{owner:?}").contains("raw_fd"));
    for (fd, expected, kind) in [
        (pidfd_for(pid), 0, PidfdObservationErrorKindV1::ExpectedPid),
        (
            pidfd_for(pid),
            pid + 1,
            PidfdObservationErrorKindV1::TargetMismatch,
        ),
        (
            File::open("/dev/null").unwrap().into(),
            pid,
            PidfdObservationErrorKindV1::InspectPidfd,
        ),
    ] {
        assert_eq!(
            ReceivedProcessPidfdV1::admit_received(fd, expected)
                .unwrap_err()
                .kind(),
            kind
        );
    }
}

#[test]
fn received_owner_rejects_changed_descriptor_flags_and_snapshot_fields() {
    let pid = std::process::id();
    let owner = ReceivedProcessPidfdV1::admit_received(pidfd_for(pid), pid).unwrap();
    rustix::io::fcntl_setfd(&owner.pidfd, rustix::io::FdFlags::empty()).unwrap();
    assert_eq!(
        owner.revalidate().unwrap_err().kind(),
        PidfdObservationErrorKindV1::CloseOnExec
    );
    rustix::io::fcntl_setfd(&owner.pidfd, rustix::io::FdFlags::CLOEXEC).unwrap();
    owner.revalidate().unwrap();
    for field in 0..8 {
        let mut owner = ReceivedProcessPidfdV1::admit_received(pidfd_for(pid), pid).unwrap();
        match field {
            0 => owner.object.device ^= 1,
            1 => owner.object.inode ^= 1,
            2 => owner.object.mode ^= 1,
            3 => owner.object.uid ^= 1,
            4 => owner.object.gid ^= 1,
            5 => owner.object.links ^= 1,
            6 => {
                owner.target.source = match owner.target.source {
                    PidfdIdentitySourceV1::KernelIoctl => PidfdIdentitySourceV1::ProcfsFdinfo,
                    PidfdIdentitySourceV1::ProcfsFdinfo => PidfdIdentitySourceV1::KernelIoctl,
                }
            }
            _ => owner.start_time_ticks += 1,
        }
        let expected = if field == 7 {
            PidfdObservationErrorKindV1::StartTimeChanged
        } else {
            PidfdObservationErrorKindV1::IdentityChanged
        };
        assert_eq!(
            owner.revalidate().unwrap_err().kind(),
            expected,
            "field {field}"
        );
    }
}

#[test]
fn received_owner_detects_descriptor_table_substitution() {
    let first = sleeping_child();
    let second = sleeping_child();
    let owner =
        ReceivedProcessPidfdV1::admit_received(pidfd_for(first.0.id()), first.0.id()).unwrap();
    let other = pidfd_for(second.0.id());
    // SAFETY: this test alone owns both descriptors and replaces the destination atomically.
    assert_eq!(
        unsafe { libc::dup3(other.as_raw_fd(), owner.pidfd.as_raw_fd(), libc::O_CLOEXEC) },
        owner.pidfd.as_raw_fd()
    );
    assert_eq!(
        owner.revalidate().unwrap_err().kind(),
        PidfdObservationErrorKindV1::IdentityChanged
    );
}

fn nonblocking_pipe() -> (OwnedFd, OwnedFd) {
    let mut raw = [-1; 2];
    // SAFETY: the array is writable for two exclusively owned descriptors returned by pipe2.
    assert_eq!(
        unsafe { libc::pipe2(raw.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) },
        0
    );
    // SAFETY: the successful syscall transferred both descriptors exclusively to this test.
    let (reader, writer) = unsafe { (OwnedFd::from_raw_fd(raw[0]), OwnedFd::from_raw_fd(raw[1])) };
    (reader, writer)
}

fn wait_for_pipe_eof(
    reader: &OwnedFd,
    timeout: Duration,
    mut pending: impl FnMut(),
) -> io::Result<()> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::other("pipe EOF deadline overflow"))?;
    loop {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "pipe EOF not observed",
            ));
        }
        match rustix::io::read(reader, &mut [0]) {
            Ok(0) => return Ok(()),
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "data before pipe EOF",
                ));
            }
            Err(error) => {
                let error = io::Error::from(error);
                if !matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) {
                    return Err(error);
                }
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "pipe EOF not observed",
            ));
        }
        // A concurrent fork can retain a CLOEXEC writer until its child execs.
        // Only a zero-byte read proves closure; readiness or a timeout does not.
        pending();
        std::thread::sleep(remaining.min(Duration::from_millis(1)));
    }
}

#[test]
fn rejected_writer_eof_waits_for_a_delayed_duplicate() {
    let (reader, writer) = nonblocking_pipe();
    let mut duplicate = Some(rustix::io::fcntl_dupfd_cloexec(&writer, 0).unwrap());
    assert!(ReceivedProcessPidfdV1::admit_received(writer, std::process::id()).is_err());
    let mut observed_pending = false;
    wait_for_pipe_eof(&reader, Duration::from_secs(5), || {
        observed_pending = true;
        drop(duplicate.take());
    })
    .unwrap();
    assert!(observed_pending);
    assert!(duplicate.is_none());
}

#[test]
fn rejected_writer_eof_refuses_a_retained_duplicate() {
    let (reader, writer) = nonblocking_pipe();
    let duplicate = rustix::io::fcntl_dupfd_cloexec(&writer, 0).unwrap();
    assert!(ReceivedProcessPidfdV1::admit_received(writer, std::process::id()).is_err());
    assert_eq!(
        wait_for_pipe_eof(&reader, Duration::from_millis(10), || {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut,
    );
    drop(duplicate);
    wait_for_pipe_eof(&reader, Duration::from_secs(5), || {}).unwrap();
}

#[test]
fn rejected_writer_eof_refuses_unexpected_data() {
    let (reader, writer) = nonblocking_pipe();
    assert_eq!(rustix::io::write(&writer, b"x").unwrap(), 1);
    assert!(ReceivedProcessPidfdV1::admit_received(writer, std::process::id()).is_err());
    assert_eq!(
        wait_for_pipe_eof(&reader, Duration::from_secs(5), || {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData,
    );
}

#[test]
fn rejected_received_descriptors_are_consumed_and_non_cloexec_is_rejected() {
    let pid = std::process::id();
    let fd = pidfd_for(pid);
    rustix::io::fcntl_setfd(&fd, rustix::io::FdFlags::empty()).unwrap();
    assert_eq!(
        ReceivedProcessPidfdV1::admit_received(fd, pid)
            .unwrap_err()
            .kind(),
        PidfdObservationErrorKindV1::CloseOnExec
    );
    let (reader, writer) = nonblocking_pipe();
    assert!(ReceivedProcessPidfdV1::admit_received(writer, pid).is_err());
    wait_for_pipe_eof(&reader, Duration::from_secs(5), || {})
        .expect("rejected consuming admission must close the original pipe writer");
}

#[test]
fn received_owner_detects_exit_without_reaping() {
    let mut child = sleeping_child();
    let owner =
        ReceivedProcessPidfdV1::admit_received(pidfd_for(child.0.id()), child.0.id()).unwrap();
    child.0.kill().unwrap();
    await_exit(&owner.pidfd);
    assert_eq!(
        owner.revalidate().unwrap_err().kind(),
        PidfdObservationErrorKindV1::AlreadyDead
    );
    assert_eq!(
        ReceivedProcessPidfdV1::admit_received(pidfd_for(child.0.id()), child.0.id())
            .unwrap_err()
            .kind(),
        PidfdObservationErrorKindV1::AlreadyDead
    );
    // A successful wait demonstrates that neither observation reaped this child.
    assert!(!child.0.wait().unwrap().success());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn received_parent_pidfd_works_with_process_control_denied() {
    const CASE: &str = "FE2O3_PIDFD_SANDBOX_TEST_FD";
    if let Some(raw) = std::env::var_os(CASE) {
        let raw = raw.to_str().unwrap().parse::<i32>().unwrap();
        // SAFETY: getppid has no arguments or process-control effects.
        let parent = unsafe { libc::getppid() } as u32;
        // SAFETY: exactly this descriptor is transferred to this isolated subprocess.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        rustix::io::fcntl_setfd(&fd, rustix::io::FdFlags::CLOEXEC).unwrap();
        let denied = [
            libc::SYS_pidfd_open,
            libc::SYS_waitid,
            libc::SYS_pidfd_send_signal,
            libc::SYS_shutdown,
            libc::SYS_clone,
            libc::SYS_clone3,
            libc::SYS_fork,
            libc::SYS_vfork,
            libc::SYS_socket,
            libc::SYS_socketpair,
            libc::SYS_connect,
            libc::SYS_bind,
        ];
        let stmt = |code: u16, k| libc::sock_filter {
            code,
            jt: 0,
            jf: 0,
            k,
        };
        let mut filters = vec![
            stmt((libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16, 4),
            libc::sock_filter {
                code: (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
                jt: 1,
                jf: 0,
                k: 0xc000003e,
            },
            stmt(
                (libc::BPF_RET | libc::BPF_K) as u16,
                libc::SECCOMP_RET_KILL_PROCESS,
            ),
            stmt((libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16, 0),
        ];
        for syscall in denied {
            filters.push(libc::sock_filter {
                code: (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
                jt: 0,
                jf: 1,
                k: syscall as u32,
            });
            filters.push(stmt(
                (libc::BPF_RET | libc::BPF_K) as u16,
                libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
            ));
        }
        filters.push(stmt(
            (libc::BPF_RET | libc::BPF_K) as u16,
            libc::SECCOMP_RET_ALLOW,
        ));
        let program = libc::sock_fprog {
            len: filters.len() as u16,
            filter: filters.as_mut_ptr(),
        };
        // SAFETY: irreversible restrictions only on this isolated helper's calling thread;
        // the program is initialized and remains borrowed during installation.
        unsafe {
            assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
            assert_eq!(
                libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &program),
                0
            );
        }
        for syscall in denied {
            // SAFETY: all calls are denied before evaluation; invalid scalar arguments would
            // also prevent accidental process-control effects.
            assert_eq!(
                unsafe { libc::syscall(syscall, -1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64) },
                -1
            );
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EPERM));
        }
        let owner = ReceivedProcessPidfdV1::admit_received(fd, parent).unwrap();
        owner.revalidate().unwrap();
        assert_eq!(owner.pid(), parent);
        return;
    }
    let pidfd = pidfd_for(std::process::id());
    let raw = pidfd.as_raw_fd();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "pidfd::owner_tests::received_parent_pidfd_works_with_process_control_denied",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CASE, raw.to_string());
    // SAFETY: only the fork child's descriptor table changes before exec; the parent retains fd.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(raw, libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = ChildGuard(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(Instant::now() < deadline, "sandbox helper timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}
