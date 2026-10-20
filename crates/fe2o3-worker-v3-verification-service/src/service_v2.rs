use std::error::Error;
use std::fmt;
use std::io::IoSliceMut;
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::time::{Duration, Instant};

use fe2o3_worker_v3_verification_protocol::{
    WORKER_V3_VERIFICATION_CURRENT_RECORD_BYTES_V2, WorkerV3VerificationChallengeFrameV2,
    WorkerV3VerificationChallengeReservationV2, WorkerV3VerificationCurrentRecordFrameV2,
    WorkerV3VerificationFdPayloadKindV1, WorkerV3VerificationProtocolErrorV2,
    WorkerV3VerificationRequestV1, WorkerV3VerificationTerminalFrameV2,
};
use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, recvmsg};

use crate::service::{
    RetainedWorkerV3VerificationPayloadV1, WorkerV3VerificationCallerV1,
    WorkerV3VerificationChallengeReplayGuardV1, WorkerV3VerificationMeasurementResolverV1,
    WorkerV3VerificationPolicyResolverV1, WorkerV3VerificationRejectionReasonV1,
    WorkerV3VerificationServiceErrorV1, caller_identity, canonical_filesystem_unix_address,
    capture_payload, object_key, receive_request, require_passcred, require_peer_write_eof,
    send_response, validate_accepted_control, validate_control, wait_for,
};

/// Required fail-closed source of service-owned current-record challenge reservations.
///
/// No implementation is provided. A production provider must atomically reserve nonzero,
/// unpredictable challenge bytes and a nonzero opaque reservation identity, durably exclude their
/// reuse across every covered process and restart, and bind release/expiry to its deployment
/// policy. The generic transport can commit and correlate those values but cannot prove those
/// persistence properties.
pub trait WorkerV3VerificationChallengeReservationProviderV2 {
    /// Reserves the sole challenge coordinate for one fully admitted Begin transaction.
    fn reserve_current_record_challenge(
        &mut self,
        caller: WorkerV3VerificationCallerV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationChallengeReservationV2>;
}

/// Stable local reason for rejecting a V2 phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkerV3VerificationRejectionReasonV2 {
    /// A V1 Begin admission policy rejected the decoded request.
    Begin(WorkerV3VerificationRejectionReasonV1),
    /// The peer queued another packet or write-half EOF before challenge release.
    BeginPhaseOrder,
    /// The injected provider did not reserve a challenge coordinate.
    ChallengeReservationUnavailable,
    /// The compiler-current-record packet violated size, credential, or ancillary rules.
    CurrentRecordTransfer,
    /// The compiler-current-record packet was not the exact canonical V2 frame.
    CurrentRecordFraming,
    /// The frame named another Begin request, challenge, or reservation identity.
    CurrentRecordAssociation,
    /// An application rejected a canonically admitted current-record submission.
    ApplicationRejected,
}

/// Terminal Begin rejection after a correlated generic rejection frame was sent.
pub struct RejectedWorkerV3VerificationBeginV2 {
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    frame: WorkerV3VerificationChallengeFrameV2,
    reason: WorkerV3VerificationRejectionReasonV2,
}

impl RejectedWorkerV3VerificationBeginV2 {
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    pub const fn frame(&self) -> &WorkerV3VerificationChallengeFrameV2 {
        &self.frame
    }

    pub const fn reason(&self) -> WorkerV3VerificationRejectionReasonV2 {
        self.reason
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for RejectedWorkerV3VerificationBeginV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedWorkerV3VerificationBeginV2")
            .field("caller", &self.caller)
            .field("request", &self.request.identity())
            .field("reason", &self.reason)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

/// Result of accepting the one V2 Begin packet.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationBeginOutcomeV2 {
    /// Immutable payload custody and one service challenge reservation are retained.
    Reserved(PendingWorkerV3VerificationCurrentRecordSessionV2),
    /// The exact decoded Begin request was rejected and the connection was closed.
    Rejected(RejectedWorkerV3VerificationBeginV2),
}

impl WorkerV3VerificationBeginOutcomeV2 {
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Move-only admitted service side of one connected filesystem-path V2 endpoint.
///
/// Admission requires exact nonblocking close-on-exec read/write custody, a connected Unix
/// `SOCK_SEQPACKET`, a local address exactly matching the caller-supplied canonical absolute
/// filesystem path, an unnamed peer address, and `SO_PASSCRED` already enabled by the listener
/// before `listen` and `accept`. It snapshots the connecting process identity from `SO_PEERCRED`;
/// every later request packet must carry matching kernel-stamped `SCM_CREDENTIALS`.
///
/// This type neither creates nor discovers a listener and grants no verification authority.
pub struct WorkerV3VerificationAcceptedServiceEndpointV2 {
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    expected_service_address: rustix::net::SocketAddrAny,
}

impl WorkerV3VerificationAcceptedServiceEndpointV2 {
    pub(crate) fn into_readiness_control(
        self,
    ) -> Result<
        (OwnedFd, WorkerV3VerificationCallerV1),
        WorkerV3VerificationAcceptedServiceAdmissionFailureV2,
    > {
        let checked = (|| {
            validate_accepted_control(&self.control, &self.expected_service_address)?;
            require_passcred(&self.control)?;
            if caller_identity(&self.control)? != self.caller {
                return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
                    "accepted control SO_PEERCRED changed after admission",
                ));
            }
            Ok(())
        })();
        match checked {
            Ok(()) => Ok((self.control, self.caller)),
            Err(source) => Err(WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
                control: self.control,
                source,
            }),
        }
    }
}

impl WorkerV3VerificationAcceptedServiceEndpointV2 {
    /// Admits one supervisor-provisioned accepted connection without receiving protocol bytes.
    ///
    /// The listener must have been prepared with
    /// [`crate::prepare_worker_v3_verification_receiver_v1`] before the client could connect.
    /// `expected_service_path` is validated lexically and matched exactly against the accepted
    /// endpoint's local address. Failure retains ownership of `control`.
    pub fn admit(
        control: OwnedFd,
        expected_service_path: &Path,
    ) -> Result<Self, WorkerV3VerificationAcceptedServiceAdmissionFailureV2> {
        let admitted = (|| {
            let expected_service_address = canonical_filesystem_unix_address(expected_service_path)
                .ok_or(WorkerV3VerificationServiceErrorV1::InvalidControl(
                    "expected service path is not a canonical absolute filesystem pathname",
                ))?;
            validate_accepted_control(&control, &expected_service_address)?;
            require_passcred(&control)?;
            let caller = caller_identity(&control)?;
            Ok((caller, expected_service_address))
        })();
        match admitted {
            Ok((caller, expected_service_address)) => Ok(Self {
                control,
                caller,
                expected_service_address,
            }),
            Err(source) => {
                Err(WorkerV3VerificationAcceptedServiceAdmissionFailureV2 { control, source })
            }
        }
    }

    /// Returns the connection-time client identity captured from `SO_PEERCRED`.
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    /// Reports that connected endpoint admission grants no verification authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for WorkerV3VerificationAcceptedServiceEndpointV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationAcceptedServiceEndpointV2")
            .field("caller", &self.caller)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

/// Failed connected-filesystem-path service admission retaining the accepted endpoint.
pub struct WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
    control: OwnedFd,
    source: WorkerV3VerificationServiceErrorV1,
}

impl WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
    pub(crate) fn into_parts(self) -> (OwnedFd, WorkerV3VerificationServiceErrorV1) {
        (self.control, self.source)
    }

    /// Returns the exact admission error.
    pub const fn source_error(&self) -> &WorkerV3VerificationServiceErrorV1 {
        &self.source
    }

    /// Returns ownership of the rejected accepted endpoint.
    pub fn into_control(self) -> OwnedFd {
        self.control
    }
}

impl fmt::Debug for WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationAcceptedServiceAdmissionFailureV2")
            .field("source", &self.source)
            .field("endpoint_retained", &true)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "V2 connected-path service admission failed with endpoint retained: {}",
            self.source
        )
    }
}

impl Error for WorkerV3VerificationAcceptedServiceAdmissionFailureV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Begins one multi-phase V2 session under one absolute deadline.
///
/// The Begin wire payload is the exact canonical V1 request, preserving its caller nonce and
/// durable replay interface. Unlike the V1 one-shot entrypoint, this function does not require EOF:
/// it captures both payloads, reserves a distinct service-owned current-record challenge, returns
/// that challenge to the client, and retains the same connection for the second phase.
pub fn begin_worker_v3_verification_session_v2<P, M, R, C>(
    control: OwnedFd,
    timeout: Duration,
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
    challenge_provider: &mut C,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
    C: WorkerV3VerificationChallengeReservationProviderV2,
{
    if timeout.is_zero() {
        return Err(WorkerV3VerificationServiceErrorV2::InvalidTimeout);
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(WorkerV3VerificationServiceErrorV2::DeadlineOverflow)?;
    begin_worker_v3_verification_session_until_v2(
        control,
        deadline,
        policy_resolver,
        measurement_resolver,
        replay_guard,
        challenge_provider,
    )
}

/// Begins one multi-phase V2 session under one exact caller-supplied deadline.
///
/// The monotonic deadline is retained unchanged for Begin admission, current-record receipt, and
/// the terminal send. An already expired or unrepresentable deadline fails before any packet or
/// descriptor is received from `control`.
pub fn begin_worker_v3_verification_session_until_v2<P, M, R, C>(
    control: OwnedFd,
    deadline: Instant,
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
    challenge_provider: &mut C,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
    C: WorkerV3VerificationChallengeReservationProviderV2,
{
    require_deadline(deadline)?;
    validate_control(&control).map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    require_passcred(&control).map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    let caller = caller_identity(&control).map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    begin_admitted_worker_v3_verification_session_until_v2(
        control,
        caller,
        deadline,
        policy_resolver,
        measurement_resolver,
        replay_guard,
        challenge_provider,
    )
}

/// Begins one V2 session on an explicitly admitted filesystem-path accepted connection.
///
/// The endpoint remains already connected; this function performs no discovery, `connect`,
/// `listen`, or `accept`. The exact caller-supplied deadline is shared by every V2 phase. Existing
/// unnamed V2 entrypoints remain unchanged and do not accept pathname endpoints.
pub fn begin_worker_v3_verification_accepted_session_until_v2<P, M, R, C>(
    endpoint: WorkerV3VerificationAcceptedServiceEndpointV2,
    deadline: Instant,
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
    challenge_provider: &mut C,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
    C: WorkerV3VerificationChallengeReservationProviderV2,
{
    require_deadline(deadline)?;
    let WorkerV3VerificationAcceptedServiceEndpointV2 {
        control,
        caller,
        expected_service_address,
    } = endpoint;
    validate_accepted_control(&control, &expected_service_address)
        .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    require_passcred(&control).map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    if caller_identity(&control).map_err(WorkerV3VerificationServiceErrorV2::V1)? != caller {
        return Err(WorkerV3VerificationServiceErrorV2::V1(
            WorkerV3VerificationServiceErrorV1::InvalidControl(
                "accepted control SO_PEERCRED changed after admission",
            ),
        ));
    }
    begin_admitted_worker_v3_verification_session_until_v2(
        control,
        caller,
        deadline,
        policy_resolver,
        measurement_resolver,
        replay_guard,
        challenge_provider,
    )
}

fn begin_admitted_worker_v3_verification_session_until_v2<P, M, R, C>(
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    deadline: Instant,
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
    challenge_provider: &mut C,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
    C: WorkerV3VerificationChallengeReservationProviderV2,
{
    let (request_bytes, descriptors) = receive_request(&control, caller, deadline)
        .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    let request = WorkerV3VerificationRequestV1::decode_canonical(&request_bytes)
        .map_err(WorkerV3VerificationServiceErrorV2::CanonicalBegin)?;
    begin_received_worker_v3_verification_session_until_v2(
        (control, caller, deadline, |control: &OwnedFd, _| {
            phase_input_is_queued(control)
        }),
        request,
        descriptors,
        policy_resolver,
        measurement_resolver,
        replay_guard,
        challenge_provider,
    )
}

pub(crate) fn begin_received_worker_v3_verification_session_until_v2<P, M, R, C, F>(
    session: (OwnedFd, WorkerV3VerificationCallerV1, Instant, F),
    request: WorkerV3VerificationRequestV1,
    descriptors: [OwnedFd; 2],
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
    challenge_provider: &mut C,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
    C: WorkerV3VerificationChallengeReservationProviderV2,
    F: FnMut(&OwnedFd, Instant) -> Result<bool, WorkerV3VerificationServiceErrorV2>,
{
    let (control, caller, deadline, mut phase_probe) = session;
    if phase_probe(&control, deadline)? {
        return reject_begin(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder,
            deadline,
        );
    }

    let expected_policy = match policy_resolver.resolve_expected_policy(caller, &request) {
        Some(policy) => policy,
        None => {
            return reject_begin_v1(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::PolicyUnresolved,
                deadline,
            );
        }
    };
    if request.policy_identity() != expected_policy {
        return reject_begin_v1(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::PolicyMismatch,
            deadline,
        );
    }
    let expected_measurement = match measurement_resolver.resolve_expected_measurement(
        caller,
        expected_policy,
        &request,
    ) {
        Some(measurement) => measurement,
        None => {
            return reject_begin_v1(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::MeasurementUnresolved,
                deadline,
            );
        }
    };
    if request.measurement_identity() != expected_measurement {
        return reject_begin_v1(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::MeasurementMismatch,
            deadline,
        );
    }
    if !replay_guard.admit_fresh_challenge(caller, expected_policy, request.challenge()) {
        return reject_begin_v1(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::ChallengeReplay,
            deadline,
        );
    }

    let [load_envelope_source, finalized_hsaco_source] = descriptors;
    let load_envelope_key = match object_key(&load_envelope_source) {
        Ok(key) => key,
        Err(_) => {
            return reject_begin_v1(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(
                    WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2,
                ),
                deadline,
            );
        }
    };
    let finalized_hsaco_key = match object_key(&finalized_hsaco_source) {
        Ok(key) => key,
        Err(_) => {
            return reject_begin_v1(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(
                    WorkerV3VerificationFdPayloadKindV1::FinalizedHsaco,
                ),
                deadline,
            );
        }
    };
    if load_envelope_key == finalized_hsaco_key {
        return reject_begin_v1(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::PayloadDescriptorAlias,
            deadline,
        );
    }
    let load_envelope = match capture_payload(load_envelope_source, request.payloads()[0], caller) {
        Ok(payload) => payload,
        Err(reason) => return reject_begin_v1(control, caller, request, reason, deadline),
    };
    let finalized_hsaco =
        match capture_payload(finalized_hsaco_source, request.payloads()[1], caller) {
            Ok(payload) => payload,
            Err(reason) => return reject_begin_v1(control, caller, request, reason, deadline),
        };
    if phase_probe(&control, deadline)? {
        return reject_begin(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder,
            deadline,
        );
    }
    let reservation = match challenge_provider.reserve_current_record_challenge(caller, &request) {
        Some(reservation) => reservation,
        None => {
            return reject_begin(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV2::ChallengeReservationUnavailable,
                deadline,
            );
        }
    };
    if phase_probe(&control, deadline)? {
        return reject_begin(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV2::BeginPhaseOrder,
            deadline,
        );
    }
    let challenge_frame = WorkerV3VerificationChallengeFrameV2::reserved(&request, &reservation);
    send_response(&control, challenge_frame.encode_canonical(), deadline)
        .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    Ok(WorkerV3VerificationBeginOutcomeV2::Reserved(
        PendingWorkerV3VerificationCurrentRecordSessionV2 {
            control,
            deadline,
            caller,
            request,
            reservation,
            payloads: [load_envelope, finalized_hsaco],
        },
    ))
}

fn reject_begin_v1(
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    reason: WorkerV3VerificationRejectionReasonV1,
    deadline: Instant,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2> {
    reject_begin(
        control,
        caller,
        request,
        WorkerV3VerificationRejectionReasonV2::Begin(reason),
        deadline,
    )
}

fn reject_begin(
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    reason: WorkerV3VerificationRejectionReasonV2,
    deadline: Instant,
) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationServiceErrorV2> {
    let frame = WorkerV3VerificationChallengeFrameV2::rejected(&request);
    send_response(&control, frame.encode_canonical(), deadline)
        .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
    drop(control);
    Ok(WorkerV3VerificationBeginOutcomeV2::Rejected(
        RejectedWorkerV3VerificationBeginV2 {
            caller,
            request,
            frame,
            reason,
        },
    ))
}

/// Pending service state retaining the one connection, reservation, and immutable payload copies.
///
/// ```compile_fail
/// use fe2o3_worker_v3_verification_service::PendingWorkerV3VerificationCurrentRecordSessionV2;
/// fn duplicate(value: PendingWorkerV3VerificationCurrentRecordSessionV2) {
///     let _again = value.clone();
/// }
/// ```
pub struct PendingWorkerV3VerificationCurrentRecordSessionV2 {
    control: OwnedFd,
    deadline: Instant,
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    reservation: WorkerV3VerificationChallengeReservationV2,
    payloads: [RetainedWorkerV3VerificationPayloadV1; 2],
}

impl fmt::Debug for PendingWorkerV3VerificationCurrentRecordSessionV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingWorkerV3VerificationCurrentRecordSessionV2")
            .field("caller", &self.caller)
            .field("request", &self.request.identity())
            .field(
                "reservation_identity",
                &self.reservation.reservation_identity(),
            )
            .field("deadline", &self.deadline)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl PendingWorkerV3VerificationCurrentRecordSessionV2 {
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    pub const fn reservation(&self) -> &WorkerV3VerificationChallengeReservationV2 {
        &self.reservation
    }

    /// Returns the exact service admission deadline retained across all remaining phases.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn payload(
        &self,
        kind: WorkerV3VerificationFdPayloadKindV1,
    ) -> &RetainedWorkerV3VerificationPayloadV1 {
        &self.payloads[kind.fd_ordinal() as usize]
    }

    /// Receives exactly one fixed-size, credential-bound current-record packet and then exact EOF.
    pub fn receive_current_record(
        self,
    ) -> Result<WorkerV3VerificationCurrentRecordOutcomeV2, WorkerV3VerificationServiceErrorV2>
    {
        let packet = receive_current_record_packet(&self.control, self.caller, self.deadline)?;
        require_peer_write_eof(&self.control, self.deadline)
            .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
        match packet {
            CurrentRecordPacketV2::Rejected(reason) => {
                Ok(WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(
                    PendingRejectedWorkerV3VerificationTerminalSessionV2 {
                        session: self,
                        reason,
                    },
                ))
            }
            CurrentRecordPacketV2::Exact(bytes) => {
                let current_record =
                    match WorkerV3VerificationCurrentRecordFrameV2::decode_canonical(&bytes) {
                        Ok(frame) => frame,
                        Err(_) => {
                            return Ok(WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(
                                PendingRejectedWorkerV3VerificationTerminalSessionV2 {
                                    session: self,
                                    reason:
                                        WorkerV3VerificationRejectionReasonV2::CurrentRecordFraming,
                                },
                            ));
                        }
                    };
                if !current_record.matches_session(&self.request, &self.reservation) {
                    return Ok(WorkerV3VerificationCurrentRecordOutcomeV2::Rejected(
                        PendingRejectedWorkerV3VerificationTerminalSessionV2 {
                            session: self,
                            reason: WorkerV3VerificationRejectionReasonV2::CurrentRecordAssociation,
                        },
                    ));
                }
                Ok(WorkerV3VerificationCurrentRecordOutcomeV2::Ready(
                    PendingWorkerV3VerificationTerminalSessionV2 {
                        session: self,
                        current_record: Box::new(current_record),
                    },
                ))
            }
        }
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Result of the compiler-current-record phase. Both variants retain immutable Begin custody.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationCurrentRecordOutcomeV2 {
    /// Exact canonical records were correlated and await an application decision.
    Ready(PendingWorkerV3VerificationTerminalSessionV2),
    /// The phase failed closed and retains enough state to send one generic rejection.
    Rejected(PendingRejectedWorkerV3VerificationTerminalSessionV2),
}

impl WorkerV3VerificationCurrentRecordOutcomeV2 {
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Application decision state retaining exact current-record and payload custody.
///
/// ```compile_fail
/// use fe2o3_worker_v3_verification_service::PendingWorkerV3VerificationTerminalSessionV2;
/// fn duplicate(value: PendingWorkerV3VerificationTerminalSessionV2) {
///     let _again = value.clone();
/// }
/// ```
pub struct PendingWorkerV3VerificationTerminalSessionV2 {
    session: PendingWorkerV3VerificationCurrentRecordSessionV2,
    current_record: Box<WorkerV3VerificationCurrentRecordFrameV2>,
}

impl fmt::Debug for PendingWorkerV3VerificationTerminalSessionV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingWorkerV3VerificationTerminalSessionV2")
            .field("caller", &self.session.caller)
            .field("request", &self.session.request.identity())
            .field("current_record", &self.current_record)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl PendingWorkerV3VerificationTerminalSessionV2 {
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.session.caller
    }

    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.session.request
    }

    pub const fn reservation(&self) -> &WorkerV3VerificationChallengeReservationV2 {
        &self.session.reservation
    }

    /// Returns the exact service admission deadline used by the terminal send.
    pub fn deadline(&self) -> Instant {
        self.session.deadline
    }

    pub fn current_record(&self) -> &WorkerV3VerificationCurrentRecordFrameV2 {
        self.current_record.as_ref()
    }

    pub fn payload(
        &self,
        kind: WorkerV3VerificationFdPayloadKindV1,
    ) -> &RetainedWorkerV3VerificationPayloadV1 {
        self.session.payload(kind)
    }

    /// Sends one bounded opaque application response and consumes the terminal capability.
    pub fn send_application_response(
        self,
        response: Vec<u8>,
    ) -> Result<CompletedWorkerV3VerificationSessionV2, WorkerV3VerificationTerminalSendFailureV2>
    {
        let frame = match WorkerV3VerificationTerminalFrameV2::application_response(
            &self.session.request,
            &self.session.reservation,
            response,
        ) {
            Ok(frame) => frame,
            Err(source) => {
                return Err(WorkerV3VerificationTerminalSendFailureV2 {
                    session: Box::new(self),
                    source: WorkerV3VerificationServiceErrorV2::Protocol(source),
                });
            }
        };
        if let Err(source) = send_response(
            &self.session.control,
            frame.encode_canonical(),
            self.session.deadline,
        ) {
            return Err(WorkerV3VerificationTerminalSendFailureV2 {
                session: Box::new(self),
                source: WorkerV3VerificationServiceErrorV2::V1(source),
            });
        }
        Ok(CompletedWorkerV3VerificationSessionV2 {
            caller: self.session.caller,
            request: self.session.request,
            reservation: self.session.reservation,
            payloads: self.session.payloads,
            current_record: Some(self.current_record),
            terminal: frame,
            rejection_reason: None,
        })
    }

    /// Sends one generic rejection chosen by the application.
    pub fn send_rejection(
        self,
    ) -> Result<CompletedWorkerV3VerificationSessionV2, WorkerV3VerificationTerminalSendFailureV2>
    {
        finish_ready_rejection(self)
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }
}

fn finish_ready_rejection(
    session: PendingWorkerV3VerificationTerminalSessionV2,
) -> Result<CompletedWorkerV3VerificationSessionV2, WorkerV3VerificationTerminalSendFailureV2> {
    let frame = WorkerV3VerificationTerminalFrameV2::rejected(
        &session.session.request,
        &session.session.reservation,
    );
    if let Err(source) = send_response(
        &session.session.control,
        frame.encode_canonical(),
        session.session.deadline,
    ) {
        return Err(WorkerV3VerificationTerminalSendFailureV2 {
            session: Box::new(session),
            source: WorkerV3VerificationServiceErrorV2::V1(source),
        });
    }
    Ok(CompletedWorkerV3VerificationSessionV2 {
        caller: session.session.caller,
        request: session.session.request,
        reservation: session.session.reservation,
        payloads: session.session.payloads,
        current_record: Some(session.current_record),
        terminal: frame,
        rejection_reason: Some(WorkerV3VerificationRejectionReasonV2::ApplicationRejected),
    })
}

/// Rejection state that still retains the reservation and both receiver-owned payload copies.
pub struct PendingRejectedWorkerV3VerificationTerminalSessionV2 {
    session: PendingWorkerV3VerificationCurrentRecordSessionV2,
    reason: WorkerV3VerificationRejectionReasonV2,
}

impl fmt::Debug for PendingRejectedWorkerV3VerificationTerminalSessionV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingRejectedWorkerV3VerificationTerminalSessionV2")
            .field("caller", &self.session.caller)
            .field("request", &self.session.request.identity())
            .field("reason", &self.reason)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl PendingRejectedWorkerV3VerificationTerminalSessionV2 {
    pub const fn reason(&self) -> WorkerV3VerificationRejectionReasonV2 {
        self.reason
    }

    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.session.request
    }

    /// Returns the exact service admission deadline used by the rejection send.
    pub fn deadline(&self) -> Instant {
        self.session.deadline
    }

    pub fn payload(
        &self,
        kind: WorkerV3VerificationFdPayloadKindV1,
    ) -> &RetainedWorkerV3VerificationPayloadV1 {
        self.session.payload(kind)
    }

    /// Sends the sole generic terminal rejection and consumes this capability.
    pub fn send_rejection(
        self,
    ) -> Result<CompletedWorkerV3VerificationSessionV2, WorkerV3VerificationRejectedSendFailureV2>
    {
        let frame = WorkerV3VerificationTerminalFrameV2::rejected(
            &self.session.request,
            &self.session.reservation,
        );
        if let Err(source) = send_response(
            &self.session.control,
            frame.encode_canonical(),
            self.session.deadline,
        ) {
            return Err(WorkerV3VerificationRejectedSendFailureV2 {
                session: Box::new(self),
                source: WorkerV3VerificationServiceErrorV2::V1(source),
            });
        }
        Ok(CompletedWorkerV3VerificationSessionV2 {
            caller: self.session.caller,
            request: self.session.request,
            reservation: self.session.reservation,
            payloads: self.session.payloads,
            current_record: None,
            terminal: frame,
            rejection_reason: Some(self.reason),
        })
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Successful terminal send retaining the complete admitted session evidence.
pub struct CompletedWorkerV3VerificationSessionV2 {
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    reservation: WorkerV3VerificationChallengeReservationV2,
    payloads: [RetainedWorkerV3VerificationPayloadV1; 2],
    current_record: Option<Box<WorkerV3VerificationCurrentRecordFrameV2>>,
    terminal: WorkerV3VerificationTerminalFrameV2,
    rejection_reason: Option<WorkerV3VerificationRejectionReasonV2>,
}

impl CompletedWorkerV3VerificationSessionV2 {
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    pub const fn reservation(&self) -> &WorkerV3VerificationChallengeReservationV2 {
        &self.reservation
    }

    pub fn payload(
        &self,
        kind: WorkerV3VerificationFdPayloadKindV1,
    ) -> &RetainedWorkerV3VerificationPayloadV1 {
        &self.payloads[kind.fd_ordinal() as usize]
    }

    pub fn current_record(&self) -> Option<&WorkerV3VerificationCurrentRecordFrameV2> {
        self.current_record.as_deref()
    }

    pub const fn terminal(&self) -> &WorkerV3VerificationTerminalFrameV2 {
        &self.terminal
    }

    pub const fn rejection_reason(&self) -> Option<WorkerV3VerificationRejectionReasonV2> {
        self.rejection_reason
    }

    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for CompletedWorkerV3VerificationSessionV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletedWorkerV3VerificationSessionV2")
            .field("caller", &self.caller)
            .field("request", &self.request.identity())
            .field("terminal", &self.terminal)
            .field("rejection_reason", &self.rejection_reason)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

/// Failed terminal response whose ready session and immutable custody are recoverable.
pub struct WorkerV3VerificationTerminalSendFailureV2 {
    session: Box<PendingWorkerV3VerificationTerminalSessionV2>,
    source: WorkerV3VerificationServiceErrorV2,
}

impl WorkerV3VerificationTerminalSendFailureV2 {
    pub const fn source_error(&self) -> &WorkerV3VerificationServiceErrorV2 {
        &self.source
    }

    pub fn into_session(self) -> PendingWorkerV3VerificationTerminalSessionV2 {
        *self.session
    }
}

impl fmt::Debug for WorkerV3VerificationTerminalSendFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationTerminalSendFailureV2")
            .field("source", &self.source)
            .field("custody_retained", &true)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WorkerV3VerificationTerminalSendFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "V2 terminal send failed with custody retained: {}",
            self.source
        )
    }
}

impl Error for WorkerV3VerificationTerminalSendFailureV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Failed rejection send whose rejection state and immutable custody are recoverable.
pub struct WorkerV3VerificationRejectedSendFailureV2 {
    session: Box<PendingRejectedWorkerV3VerificationTerminalSessionV2>,
    source: WorkerV3VerificationServiceErrorV2,
}

impl WorkerV3VerificationRejectedSendFailureV2 {
    pub const fn source_error(&self) -> &WorkerV3VerificationServiceErrorV2 {
        &self.source
    }

    pub fn into_session(self) -> PendingRejectedWorkerV3VerificationTerminalSessionV2 {
        *self.session
    }
}

impl fmt::Debug for WorkerV3VerificationRejectedSendFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationRejectedSendFailureV2")
            .field("source", &self.source)
            .field("custody_retained", &true)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WorkerV3VerificationRejectedSendFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "V2 rejection send failed with custody retained: {}",
            self.source
        )
    }
}

impl Error for WorkerV3VerificationRejectedSendFailureV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

enum CurrentRecordPacketV2 {
    Exact(Vec<u8>),
    Rejected(WorkerV3VerificationRejectionReasonV2),
}

#[repr(align(16))]
struct AlignedAncillaryStorageV2<const N: usize>([MaybeUninit<u8>; N]);

fn receive_current_record_packet(
    control: &OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    deadline: Instant,
) -> Result<CurrentRecordPacketV2, WorkerV3VerificationServiceErrorV2> {
    loop {
        wait_for(control, PollFlags::IN, deadline)
            .map_err(WorkerV3VerificationServiceErrorV2::V1)?;
        let mut payload = vec![0_u8; WORKER_V3_VERIFICATION_CURRENT_RECORD_BYTES_V2 + 1];
        let received = {
            let mut vectors = [IoSliceMut::new(&mut payload)];
            let mut space = AlignedAncillaryStorageV2(
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1), ScmCredentials(1))],
            );
            let mut ancillary = RecvAncillaryBuffer::new(&mut space.0);
            match recvmsg(
                control,
                &mut vectors,
                &mut ancillary,
                RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC | RecvFlags::TRUNC,
            ) {
                Ok(received) => {
                    let mut invalid_ancillary = false;
                    let mut credentials = None;
                    for message in ancillary.drain() {
                        match message {
                            RecvAncillaryMessage::ScmCredentials(received) => {
                                if credentials.replace(received).is_some() {
                                    invalid_ancillary = true;
                                }
                            }
                            RecvAncillaryMessage::ScmRights(received) => {
                                invalid_ancillary = true;
                                drop(received);
                            }
                            _ => invalid_ancillary = true,
                        }
                    }
                    if received.bytes == 0 && credentials.is_none() {
                        return Err(WorkerV3VerificationServiceErrorV2::PeerClosed);
                    }
                    let matching_credentials = credentials.is_some_and(|credentials| {
                        u32::try_from(credentials.pid.as_raw_pid()).ok() == Some(caller.pid())
                            && credentials.uid.as_raw() == caller.uid()
                            && credentials.gid.as_raw() == caller.gid()
                    });
                    if received
                        .flags
                        .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
                        || invalid_ancillary
                        || !matching_credentials
                        || received.bytes != WORKER_V3_VERIFICATION_CURRENT_RECORD_BYTES_V2
                    {
                        CurrentRecordPacketV2::Rejected(
                            WorkerV3VerificationRejectionReasonV2::CurrentRecordTransfer,
                        )
                    } else {
                        payload.truncate(received.bytes);
                        CurrentRecordPacketV2::Exact(payload)
                    }
                }
                Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
                Err(source) => {
                    return Err(WorkerV3VerificationServiceErrorV2::Io {
                        operation: "receive V2 current-record frame",
                        source: source.into(),
                    });
                }
            }
        };
        return Ok(received);
    }
}

fn phase_input_is_queued(control: &OwnedFd) -> Result<bool, WorkerV3VerificationServiceErrorV2> {
    let timeout = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let mut descriptors = [PollFd::new(
        control,
        PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP,
    )];
    match poll(&mut descriptors, Some(&timeout)) {
        Ok(0) => Ok(false),
        Ok(_) => Ok(descriptors[0]
            .revents()
            .intersects(PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP)),
        Err(rustix::io::Errno::INTR) => phase_input_is_queued(control),
        Err(source) => Err(WorkerV3VerificationServiceErrorV2::Io {
            operation: "inspect V2 phase ordering",
            source: source.into(),
        }),
    }
}

pub(crate) fn require_deadline(
    deadline: Instant,
) -> Result<(), WorkerV3VerificationServiceErrorV2> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(WorkerV3VerificationServiceErrorV2::V1(
            WorkerV3VerificationServiceErrorV1::Timeout,
        ));
    }
    i64::try_from(remaining.as_secs())
        .map(|_| ())
        .map_err(|_| WorkerV3VerificationServiceErrorV2::DeadlineOverflow)
}

/// Stable V2 setup or transport failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationServiceErrorV2 {
    InvalidTimeout,
    DeadlineOverflow,
    CanonicalBegin(fe2o3_worker_v3_verification_protocol::WorkerV3VerificationProtocolErrorV1),
    Protocol(WorkerV3VerificationProtocolErrorV2),
    V1(WorkerV3VerificationServiceErrorV1),
    PeerClosed,
    Io {
        operation: &'static str,
        source: std::io::Error,
    },
}

impl fmt::Display for WorkerV3VerificationServiceErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeout => formatter.write_str("V2 service timeout must be nonzero"),
            Self::DeadlineOverflow => formatter.write_str("V2 absolute deadline overflowed"),
            Self::CanonicalBegin(source) => write!(formatter, "V2 Begin is noncanonical: {source}"),
            Self::Protocol(source) => write!(formatter, "V2 phase framing failed: {source}"),
            Self::V1(source) => write!(formatter, "V2 shared transport failed: {source}"),
            Self::PeerClosed => formatter.write_str("V2 peer closed before current-record phase"),
            Self::Io { operation, source } => {
                write!(formatter, "V2 operation `{operation}` failed: {source}")
            }
        }
    }
}

impl Error for WorkerV3VerificationServiceErrorV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CanonicalBegin(source) => Some(source),
            Self::Protocol(source) => Some(source),
            Self::V1(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<WorkerV3VerificationProtocolErrorV2> for WorkerV3VerificationServiceErrorV2 {
    fn from(source: WorkerV3VerificationProtocolErrorV2) -> Self {
        Self::Protocol(source)
    }
}
