//! Explicit V3 readiness on an admitted pathname connection.
use crate::service::{receive_request_sized, send_response};
use crate::service_v2::{begin_received_worker_v3_verification_session_until_v2, require_deadline};
use crate::{
    WorkerV3VerificationAcceptedServiceEndpointV2, WorkerV3VerificationBeginOutcomeV2,
    WorkerV3VerificationCallerV1, WorkerV3VerificationChallengeReplayGuardV1,
    WorkerV3VerificationChallengeReservationProviderV2, WorkerV3VerificationMeasurementResolverV1,
    WorkerV3VerificationPolicyResolverV1, WorkerV3VerificationServiceErrorV1,
    WorkerV3VerificationServiceErrorV2,
};
use fe2o3_worker_v3_verification_protocol::{
    MAX_WORKER_V3_VERIFICATION_BEGIN_BYTES_V3, WorkerV3VerificationProtocolErrorV3,
    WorkerV3VerificationReadyFrameV3,
};
use std::{fmt, io, os::fd::OwnedFd, time::Instant};

/// A move-only connection which has emitted Ready only after both strict checks.
///
/// Its descriptor and nonce are private. Readiness grants no verification authority.
/// ```compile_fail
/// use fe2o3_worker_v3_verification_service::WorkerV3VerificationReadyServiceSessionV3;
/// fn duplicate(x: WorkerV3VerificationReadyServiceSessionV3) { let _ = x.clone(); }
/// ```
pub struct WorkerV3VerificationReadyServiceSessionV3 {
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    deadline: Instant,
    ready: WorkerV3VerificationReadyFrameV3,
}
impl WorkerV3VerificationReadyServiceSessionV3 {
    /// Performs the second strict endpoint check and emits one bounded ReadyV3 packet.
    ///
    /// The endpoint must already have passed V2 admission. There is no discovery, Hello,
    /// downgrade, sleep, weak entropy fallback or fresh deadline at any subsequent phase.
    pub fn prepare_until(
        endpoint: WorkerV3VerificationAcceptedServiceEndpointV2,
        deadline: Instant,
    ) -> Result<Self, WorkerV3VerificationReadinessFailureV3> {
        let (control, caller) = match endpoint.into_readiness_control() {
            Ok(parts) => parts,
            Err(failure) => {
                let (control, source) = failure.into_parts();
                return Err(cleanup_failure(
                    WorkerV3VerificationServiceErrorV3::V1(source),
                    control,
                    false,
                ));
            }
        };
        Self::prepare_checked_until(control, caller, deadline, |nonce| {
            rustix::rand::getrandom(nonce, rustix::rand::GetRandomFlags::NONBLOCK)
                .map_err(io::Error::from)
        })
    }

    fn prepare_checked_until(
        control: OwnedFd,
        caller: WorkerV3VerificationCallerV1,
        deadline: Instant,
        fill_once: impl FnOnce(&mut [u8; 32]) -> io::Result<usize>,
    ) -> Result<Self, WorkerV3VerificationReadinessFailureV3> {
        let mut attempted = false;
        let prepared = (|| {
            require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
            if input_is_queued(&control, deadline)? {
                return Err(WorkerV3VerificationServiceErrorV3::EarlyInput);
            }
            let ready = fresh_ready_with(deadline, fill_once)?;
            require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
            if input_is_queued(&control, deadline)? {
                return Err(WorkerV3VerificationServiceErrorV3::EarlyInput);
            }
            attempted = true;
            send_response(&control, &ready.encode_canonical(), deadline)
                .map_err(WorkerV3VerificationServiceErrorV3::V1)?;
            require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
            Ok(ready)
        })();
        match prepared {
            Ok(ready) => Ok(Self {
                control,
                caller,
                deadline,
                ready,
            }),
            Err(source) => {
                // A potentially delivered challenge must never leave a reusable connection.
                Err(cleanup_failure(source, control, attempted))
            }
        }
    }
    /// Retains the exact absolute deadline supplied at preparation.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }
    /// Returns the connection-time caller identity.
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }
    /// Readiness never grants verification or execution authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
    /// Consumes this connection and nonce once, then reuses the existing V2 application core.
    ///
    /// Every error closes the owned endpoint and any received payload descriptors.
    pub fn begin<P, M, R, C>(
        self,
        policy_resolver: &mut P,
        measurement_resolver: &mut M,
        replay_guard: &mut R,
        challenge_provider: &mut C,
    ) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV3>
    where
        P: WorkerV3VerificationPolicyResolverV1,
        M: WorkerV3VerificationMeasurementResolverV1,
        R: WorkerV3VerificationChallengeReplayGuardV1,
        C: WorkerV3VerificationChallengeReservationProviderV2,
    {
        require_deadline(self.deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
        let (bytes, payloads) = receive_request_sized(
            &self.control,
            self.caller,
            self.deadline,
            MAX_WORKER_V3_VERIFICATION_BEGIN_BYTES_V3,
        )
        .map_err(WorkerV3VerificationServiceErrorV3::V1)?;
        let request = self
            .ready
            .decode_begin(&bytes)
            .map_err(WorkerV3VerificationServiceErrorV3::Protocol)?;
        begin_received_worker_v3_verification_session_until_v2(
            (
                self.control,
                self.caller,
                self.deadline,
                phase_input_is_queued_until,
            ),
            request,
            payloads,
            policy_resolver,
            measurement_resolver,
            replay_guard,
            challenge_provider,
        )
        .map_err(WorkerV3VerificationServiceErrorV3::V2)
    }
}
impl fmt::Debug for WorkerV3VerificationReadyServiceSessionV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerV3VerificationReadyServiceSessionV3")
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}
/// Preparation refusal retaining only cleanup custody, never a readiness capability.
pub struct WorkerV3VerificationReadinessFailureV3 {
    source: WorkerV3VerificationServiceErrorV3,
    control: Option<OwnedFd>,
    ready_send_attempted: bool,
}
impl WorkerV3VerificationReadinessFailureV3 {
    /// Borrows the refusal reason.
    pub const fn source_error(&self) -> &WorkerV3VerificationServiceErrorV3 {
        &self.source
    }
    /// Whether a Ready send was attempted, including an unknown/partial send outcome.
    pub const fn ready_send_attempted(&self) -> bool {
        self.ready_send_attempted
    }
    /// Takes cleanup-only custody. Every retained endpoint has been shut down, even before I/O.
    pub fn into_control(self) -> Option<OwnedFd> {
        self.control
    }
}
impl fmt::Debug for WorkerV3VerificationReadinessFailureV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerV3VerificationReadinessFailureV3")
            .field("source", &self.source)
            .field("endpoint_retained", &self.control.is_some())
            .field("ready_send_attempted", &self.ready_send_attempted)
            .finish()
    }
}
/// V3 transport or framing refusal; it is never an authority-bearing result.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationServiceErrorV3 {
    /// Existing guarded transfer failure.
    V1(WorkerV3VerificationServiceErrorV1),
    /// Existing bounded session failure.
    V2(WorkerV3VerificationServiceErrorV2),
    /// Canonical framing or per-connection nonce refusal.
    Protocol(WorkerV3VerificationProtocolErrorV3),
    /// Secure nonblocking entropy failed or was incomplete.
    Entropy(io::Error),
    /// Data or EOF was already queued before readiness.
    EarlyInput,
}
impl fmt::Display for WorkerV3VerificationServiceErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V3 readiness refusal: {self:?}")
    }
}
impl std::error::Error for WorkerV3VerificationServiceErrorV3 {}

fn fresh_ready_with(
    deadline: Instant,
    fill_once: impl FnOnce(&mut [u8; 32]) -> io::Result<usize>,
) -> Result<WorkerV3VerificationReadyFrameV3, WorkerV3VerificationServiceErrorV3> {
    require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
    let mut nonce = [0; 32];
    let count = fill_once(&mut nonce).map_err(WorkerV3VerificationServiceErrorV3::Entropy)?;
    require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
    if count != nonce.len() {
        return Err(WorkerV3VerificationServiceErrorV3::Entropy(
            io::Error::other("short readiness entropy"),
        ));
    }
    WorkerV3VerificationReadyFrameV3::new(nonce)
        .map_err(WorkerV3VerificationServiceErrorV3::Protocol)
}

fn cleanup_failure(
    source: WorkerV3VerificationServiceErrorV3,
    control: OwnedFd,
    ready_send_attempted: bool,
) -> WorkerV3VerificationReadinessFailureV3 {
    // Shutdown affects aliases too: cleanup custody must not authorize a retry.
    let control = if rustix::net::shutdown(&control, rustix::net::Shutdown::Both).is_ok() {
        Some(control)
    } else {
        None
    };
    WorkerV3VerificationReadinessFailureV3 {
        source,
        control,
        ready_send_attempted,
    }
}

fn input_is_queued(
    control: &OwnedFd,
    deadline: Instant,
) -> Result<bool, WorkerV3VerificationServiceErrorV3> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    loop {
        require_deadline(deadline).map_err(WorkerV3VerificationServiceErrorV3::V2)?;
        let mut fds = [PollFd::new(
            control,
            PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP,
        )];
        match poll(
            &mut fds,
            Some(&Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        ) {
            Ok(0) => return Ok(false),
            Ok(_) => return Ok(!fds[0].revents().is_empty()),
            Err(rustix::io::Errno::INTR) => {}
            Err(e) => {
                return Err(WorkerV3VerificationServiceErrorV3::V1(
                    WorkerV3VerificationServiceErrorV1::Io {
                        operation: "inspect V3 readiness ordering",
                        source: e.into(),
                    },
                ));
            }
        }
    }
}

// V3 alone selects this policy at every shared post-Begin phase boundary.
fn phase_input_is_queued_until(
    control: &OwnedFd,
    deadline: Instant,
) -> Result<bool, WorkerV3VerificationServiceErrorV2> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    phase_probe_with(deadline, Instant::now, || {
        let mut fds = [PollFd::new(
            control,
            PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP,
        )];
        poll(
            &mut fds,
            Some(&Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        )
        .map(|count| count != 0 && !fds[0].revents().is_empty())
    })
}

fn phase_probe_with(
    deadline: Instant,
    mut now: impl FnMut() -> Instant,
    mut poll_once: impl FnMut() -> Result<bool, rustix::io::Errno>,
) -> Result<bool, WorkerV3VerificationServiceErrorV2> {
    loop {
        if now() >= deadline {
            return Err(WorkerV3VerificationServiceErrorV2::V1(
                WorkerV3VerificationServiceErrorV1::Timeout,
            ));
        }
        // Keep the same overflow refusal as the existing deadline validator.
        crate::service_v2::require_deadline(deadline)?;
        let result = poll_once();
        if now() >= deadline {
            return Err(WorkerV3VerificationServiceErrorV2::V1(
                WorkerV3VerificationServiceErrorV1::Timeout,
            ));
        }
        match result {
            Ok(queued) => return Ok(queued),
            Err(rustix::io::Errno::INTR) => {}
            Err(source) => {
                return Err(WorkerV3VerificationServiceErrorV2::Io {
                    operation: "inspect V3 phase ordering",
                    source: source.into(),
                });
            }
        }
    }
}

#[cfg(test)]
#[path = "service_v3/tests.rs"]
mod readiness_phase_tests;

#[cfg(test)]
mod readiness_entropy_tests {
    use super::*;
    #[test]
    fn zero_and_short_entropy_are_terminal_errors() {
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        assert!(matches!(
            fresh_ready_with(deadline, |_| Ok(32)),
            Err(WorkerV3VerificationServiceErrorV3::Protocol(
                WorkerV3VerificationProtocolErrorV3::ZeroNonce
            ))
        ));
        assert!(matches!(
            fresh_ready_with(deadline, |nonce| {
                nonce.fill(1);
                Ok(31)
            }),
            Err(WorkerV3VerificationServiceErrorV3::Entropy(_))
        ));
    }
    #[test]
    fn interrupted_entropy_is_not_retried_or_replaced() {
        let calls = std::cell::Cell::new(0);
        let err = fresh_ready_with(Instant::now() + std::time::Duration::from_secs(1), |_| {
            calls.set(calls.get() + 1);
            Err(io::Error::from(io::ErrorKind::Interrupted))
        })
        .unwrap_err();
        assert!(matches!(
            err,
            WorkerV3VerificationServiceErrorV3::Entropy(_)
        ));
        assert_eq!(calls.get(), 1);
    }
    #[test]
    fn preexpired_deadline_never_acquires_entropy() {
        assert!(
            fresh_ready_with(Instant::now(), |_| panic!("expired entropy acquisition")).is_err()
        );
    }
}
