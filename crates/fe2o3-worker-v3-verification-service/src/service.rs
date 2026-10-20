use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, IoSliceMut, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::time::{Duration, Instant};

use fe2o3_worker_v3_verification_protocol::{
    MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1, WORKER_V3_VERIFICATION_FD_PAYLOADS_V1,
    WorkerV3VerificationFdPayloadDescriptorV1, WorkerV3VerificationFdPayloadKindV1,
    WorkerV3VerificationFreshChallengeV1, WorkerV3VerificationMeasurementIdentityV1,
    WorkerV3VerificationPolicyIdentityV1, WorkerV3VerificationProtocolErrorV1,
    WorkerV3VerificationRequestV1, WorkerV3VerificationResponseDispositionV1,
    WorkerV3VerificationResponseV1, WorkerV3VerificationTranscriptIdentityV1,
};
use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::fs::{FileType, MemfdFlags, Mode, OFlags, SealFlags};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendFlags,
    SocketAddrAny, SocketAddrUnix, SocketType, recvmsg,
};
use sha2::{Digest, Sha256};

const PAYLOAD_COPY_CHUNK_BYTES_V1: usize = 64 * 1024;
const TMPFS_MAGIC_V1: u64 = 0x0102_1994;
const LINUX_MAX_CANONICAL_FILESYSTEM_SOCKET_PATH_BYTES: usize = 107;
const TRANSCRIPT_DOMAIN_V1: &[u8] = b"FE2O3/WORKER-V3/VERIFICATION-SERVICE-TRANSCRIPT/V1\0";
const REQUIRED_IMMUTABLE_SEALS_V1: SealFlags = SealFlags::WRITE
    .union(SealFlags::GROW)
    .union(SealFlags::SHRINK)
    .union(SealFlags::SEAL);

#[repr(align(16))]
struct AlignedAncillaryStorageV1<const N: usize>([MaybeUninit<u8>; N]);

/// Kernel-reported identity of the peer that submitted one session packet.
///
/// `SO_PEERCRED` is a connection-time identity. It does not prove process liveness, executable
/// measurement, or authorization; the injected resolvers must apply the deployment's caller
/// policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkerV3VerificationCallerV1 {
    pid: u32,
    uid: u32,
    gid: u32,
}

impl WorkerV3VerificationCallerV1 {
    /// Returns the positive kernel-reported peer PID.
    pub const fn pid(self) -> u32 {
        self.pid
    }

    /// Returns the kernel-reported peer UID.
    pub const fn uid(self) -> u32 {
        self.uid
    }

    /// Returns the kernel-reported peer GID.
    pub const fn gid(self) -> u32 {
        self.gid
    }
}

/// Resolves the only policy identity allowed for one caller and request.
///
/// Returning `None` rejects the request. Production implementations must authenticate and pin
/// their policy store; this boundary deliberately provides no permissive default.
pub trait WorkerV3VerificationPolicyResolverV1 {
    /// Returns the expected complete policy identity or rejects the caller/request pair.
    fn resolve_expected_policy(
        &mut self,
        caller: WorkerV3VerificationCallerV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationPolicyIdentityV1>;
}

/// Resolves the only verifier measurement allowed for one caller, policy, and request.
///
/// Returning `None` rejects the request. The identity remains a caller/deployment selection; this
/// trait does not measure a process and cannot establish verifier authenticity.
pub trait WorkerV3VerificationMeasurementResolverV1 {
    /// Returns the expected verifier measurement identity or rejects the session.
    fn resolve_expected_measurement(
        &mut self,
        caller: WorkerV3VerificationCallerV1,
        policy: WorkerV3VerificationPolicyIdentityV1,
        request: &WorkerV3VerificationRequestV1,
    ) -> Option<WorkerV3VerificationMeasurementIdentityV1>;
}

/// Atomically excludes reuse of caller-generated protocol challenges.
///
/// Returning `false` rejects the request. A production implementation must durably and atomically
/// retain admitted `(caller, policy, challenge)` tuples across every process and restart covered by
/// the policy. An in-memory or check-then-insert implementation is not sufficient for production.
pub trait WorkerV3VerificationChallengeReplayGuardV1 {
    /// Atomically consumes a fresh challenge exactly once.
    fn admit_fresh_challenge(
        &mut self,
        caller: WorkerV3VerificationCallerV1,
        policy: WorkerV3VerificationPolicyIdentityV1,
        challenge: WorkerV3VerificationFreshChallengeV1,
    ) -> bool;
}

/// Stable local reason for an authority-free request rejection.
///
/// The wire response intentionally does not reveal this reason. Every variant remains a framing
/// rejection and grants no theorem, load, or launch authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkerV3VerificationRejectionReasonV1 {
    /// No policy was resolved for the caller and request.
    PolicyUnresolved,
    /// The request named a policy other than the resolved caller policy.
    PolicyMismatch,
    /// No expected verifier measurement was resolved.
    MeasurementUnresolved,
    /// The request named a verifier measurement other than the resolved expectation.
    MeasurementMismatch,
    /// The caller challenge was already consumed or could not be atomically admitted.
    ChallengeReplay,
    /// The two payload descriptors refer to the same underlying object.
    PayloadDescriptorAlias,
    /// A payload was not an exact read-only, close-on-exec, sealed anonymous tmpfs file.
    InvalidPayloadDescriptor(WorkerV3VerificationFdPayloadKindV1),
    /// A payload object length differed from its canonical request descriptor.
    PayloadLengthMismatch(WorkerV3VerificationFdPayloadKindV1),
    /// A payload could not be read to the exact declared EOF.
    PayloadBoundaryMismatch(WorkerV3VerificationFdPayloadKindV1),
    /// A payload SHA-256 differed from its canonical request descriptor.
    PayloadDigestMismatch(WorkerV3VerificationFdPayloadKindV1),
    /// Receiver-owned immutable snapshot creation failed closed.
    PayloadCustodyFailed(WorkerV3VerificationFdPayloadKindV1),
}

impl WorkerV3VerificationRejectionReasonV1 {
    const fn transcript_tag(self) -> u16 {
        match self {
            Self::PolicyUnresolved => 1,
            Self::PolicyMismatch => 2,
            Self::MeasurementUnresolved => 3,
            Self::MeasurementMismatch => 4,
            Self::ChallengeReplay => 5,
            Self::PayloadDescriptorAlias => 6,
            Self::InvalidPayloadDescriptor(kind) => 10 + payload_kind_tag(kind),
            Self::PayloadLengthMismatch(kind) => 20 + payload_kind_tag(kind),
            Self::PayloadBoundaryMismatch(kind) => 30 + payload_kind_tag(kind),
            Self::PayloadDigestMismatch(kind) => 40 + payload_kind_tag(kind),
            Self::PayloadCustodyFailed(kind) => 50 + payload_kind_tag(kind),
        }
    }
}

const fn payload_kind_tag(kind: WorkerV3VerificationFdPayloadKindV1) -> u16 {
    match kind {
        WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2 => 1,
        WorkerV3VerificationFdPayloadKindV1::FinalizedHsaco => 2,
        _ => 0,
    }
}

/// Receiver-owned immutable copy of one request payload.
///
/// Descriptor access exposes exact bytes only. This value is not protected verification evidence
/// and cannot grant load or launch authority.
pub struct RetainedWorkerV3VerificationPayloadV1 {
    file: File,
    kind: WorkerV3VerificationFdPayloadKindV1,
    byte_len: u64,
    sha256: [u8; 32],
}

impl fmt::Debug for RetainedWorkerV3VerificationPayloadV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedWorkerV3VerificationPayloadV1")
            .field("kind", &self.kind)
            .field("byte_len", &self.byte_len)
            .field("sha256", &self.sha256)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl AsFd for RetainedWorkerV3VerificationPayloadV1 {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl RetainedWorkerV3VerificationPayloadV1 {
    /// Returns the canonical payload role.
    pub const fn kind(&self) -> WorkerV3VerificationFdPayloadKindV1 {
        self.kind
    }

    /// Returns the exact byte length observed during immutable capture.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Returns the exact SHA-256 observed during immutable capture.
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }

    /// Reports that immutable byte custody is not verification authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Fully framed, caller-bound, replay-admitted session with immutable payload custody.
///
/// This state means only that transport, selection, freshness, and bytes were admitted. No
/// verifier ran and no compiler or machine theorem was established.
pub struct FramedWorkerV3VerificationSessionV1 {
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    response: WorkerV3VerificationResponseV1,
    payloads: [RetainedWorkerV3VerificationPayloadV1; WORKER_V3_VERIFICATION_FD_PAYLOADS_V1],
}

impl fmt::Debug for FramedWorkerV3VerificationSessionV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FramedWorkerV3VerificationSessionV1")
            .field("caller", &self.caller)
            .field("request", &self.request.identity())
            .field("response", &self.response.identity())
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl FramedWorkerV3VerificationSessionV1 {
    /// Returns the kernel-reported submitting peer.
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    /// Returns the exact canonical request.
    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    /// Returns the framing-only response sent to the peer.
    pub const fn response(&self) -> &WorkerV3VerificationResponseV1 {
        &self.response
    }

    /// Returns the retained payload at its canonical fd ordinal.
    pub fn payload(
        &self,
        kind: WorkerV3VerificationFdPayloadKindV1,
    ) -> &RetainedWorkerV3VerificationPayloadV1 {
        &self.payloads[kind.fd_ordinal() as usize]
    }

    /// Reports that no protected verifier was executed by this boundary.
    pub const fn verification_performed(&self) -> bool {
        false
    }

    /// Reports that this session grants no theorem, load, or launch authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Locally classified authority-free rejection whose generic response was sent to the peer.
pub struct RejectedWorkerV3VerificationSessionV1 {
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    response: WorkerV3VerificationResponseV1,
    reason: WorkerV3VerificationRejectionReasonV1,
}

impl fmt::Debug for RejectedWorkerV3VerificationSessionV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedWorkerV3VerificationSessionV1")
            .field("caller", &self.caller)
            .field("request", &self.request.identity())
            .field("response", &self.response.identity())
            .field("reason", &self.reason)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl RejectedWorkerV3VerificationSessionV1 {
    /// Returns the kernel-reported submitting peer.
    pub const fn caller(&self) -> WorkerV3VerificationCallerV1 {
        self.caller
    }

    /// Returns the exact canonical request that was rejected.
    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    /// Returns the generic framing rejection sent to the peer.
    pub const fn response(&self) -> &WorkerV3VerificationResponseV1 {
        &self.response
    }

    /// Returns the local fail-closed reason, which is not disclosed on the wire.
    pub const fn reason(&self) -> WorkerV3VerificationRejectionReasonV1 {
        self.reason
    }

    /// Reports that this rejection grants no authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Terminal outcome after exactly one service exchange.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationSessionOutcomeV1 {
    /// Transport, selection, replay exclusion, and immutable payload custody succeeded.
    Framed(FramedWorkerV3VerificationSessionV1),
    /// A decoded request failed a local admission check and received a generic rejection.
    Rejected(RejectedWorkerV3VerificationSessionV1),
}

impl WorkerV3VerificationSessionOutcomeV1 {
    /// Reports that neither possible outcome grants authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Serves exactly one bounded authority-free Worker V3 verification session.
///
/// The peer must send one packet and then shut down its write half. The function requires exact
/// write-half EOF before processing and consumes and closes `control` after sending one
/// framing-only response. Malformed transport or noncanonical request bytes terminate without a
/// response because there is no exact request identity to bind. A decoded request receives
/// `RequestRejected` for every subsequent failure. `RequestFramed` means only that immutable
/// payload custody is ready for a separate protected verifier.
pub fn serve_worker_v3_verification_session_v1<P, M, R>(
    control: OwnedFd,
    timeout: Duration,
    policy_resolver: &mut P,
    measurement_resolver: &mut M,
    replay_guard: &mut R,
) -> Result<WorkerV3VerificationSessionOutcomeV1, WorkerV3VerificationServiceErrorV1>
where
    P: WorkerV3VerificationPolicyResolverV1,
    M: WorkerV3VerificationMeasurementResolverV1,
    R: WorkerV3VerificationChallengeReplayGuardV1,
{
    if timeout.is_zero() {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidTimeout);
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(WorkerV3VerificationServiceErrorV1::DeadlineOverflow)?;
    validate_control(&control)?;
    require_passcred(&control)?;
    let caller = caller_identity(&control)?;
    let (request_bytes, descriptors) = receive_request(&control, caller, deadline)?;
    require_peer_write_eof(&control, deadline)?;
    let request = WorkerV3VerificationRequestV1::decode_canonical(&request_bytes)
        .map_err(WorkerV3VerificationServiceErrorV1::CanonicalRequest)?;

    let expected_policy = match policy_resolver.resolve_expected_policy(caller, &request) {
        Some(policy) => policy,
        None => {
            return reject(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::PolicyUnresolved,
                deadline,
            );
        }
    };
    if request.policy_identity() != expected_policy {
        return reject(
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
            return reject(
                control,
                caller,
                request,
                WorkerV3VerificationRejectionReasonV1::MeasurementUnresolved,
                deadline,
            );
        }
    };
    if request.measurement_identity() != expected_measurement {
        return reject(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::MeasurementMismatch,
            deadline,
        );
    }
    if !replay_guard.admit_fresh_challenge(caller, expected_policy, request.challenge()) {
        return reject(
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
            return reject(
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
            return reject(
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
        return reject(
            control,
            caller,
            request,
            WorkerV3VerificationRejectionReasonV1::PayloadDescriptorAlias,
            deadline,
        );
    }
    let load_envelope = match capture_payload(load_envelope_source, request.payloads()[0], caller) {
        Ok(payload) => payload,
        Err(reason) => return reject(control, caller, request, reason, deadline),
    };
    let finalized_hsaco =
        match capture_payload(finalized_hsaco_source, request.payloads()[1], caller) {
            Ok(payload) => payload,
            Err(reason) => return reject(control, caller, request, reason, deadline),
        };
    let response = make_response(
        &request,
        caller,
        WorkerV3VerificationResponseDispositionV1::RequestFramed,
        None,
    )?;
    send_response(&control, response.encode_canonical(), deadline)?;
    Ok(WorkerV3VerificationSessionOutcomeV1::Framed(
        FramedWorkerV3VerificationSessionV1 {
            caller,
            request,
            response,
            payloads: [load_envelope, finalized_hsaco],
        },
    ))
}

fn reject(
    control: OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    request: WorkerV3VerificationRequestV1,
    reason: WorkerV3VerificationRejectionReasonV1,
    deadline: Instant,
) -> Result<WorkerV3VerificationSessionOutcomeV1, WorkerV3VerificationServiceErrorV1> {
    let response = make_response(
        &request,
        caller,
        WorkerV3VerificationResponseDispositionV1::RequestRejected,
        Some(reason),
    )?;
    send_response(&control, response.encode_canonical(), deadline)?;
    Ok(WorkerV3VerificationSessionOutcomeV1::Rejected(
        RejectedWorkerV3VerificationSessionV1 {
            caller,
            request,
            response,
            reason,
        },
    ))
}

fn make_response(
    request: &WorkerV3VerificationRequestV1,
    caller: WorkerV3VerificationCallerV1,
    disposition: WorkerV3VerificationResponseDispositionV1,
    reason: Option<WorkerV3VerificationRejectionReasonV1>,
) -> Result<WorkerV3VerificationResponseV1, WorkerV3VerificationServiceErrorV1> {
    let mut digest = Sha256::new();
    digest.update(TRANSCRIPT_DOMAIN_V1);
    digest.update(1_u16.to_le_bytes());
    digest.update(caller.pid.to_le_bytes());
    digest.update(caller.uid.to_le_bytes());
    digest.update(caller.gid.to_le_bytes());
    digest.update(request.identity().as_bytes());
    digest.update(
        match disposition {
            WorkerV3VerificationResponseDispositionV1::RequestFramed => 1_u16,
            WorkerV3VerificationResponseDispositionV1::RequestRejected => 2_u16,
            _ => 0_u16,
        }
        .to_le_bytes(),
    );
    digest.update(
        reason
            .map_or(0, WorkerV3VerificationRejectionReasonV1::transcript_tag)
            .to_le_bytes(),
    );
    let transcript = WorkerV3VerificationTranscriptIdentityV1::new(digest.finalize().into())
        .map_err(|_| WorkerV3VerificationServiceErrorV1::TranscriptIdentity)?;
    Ok(WorkerV3VerificationResponseV1::new(
        request,
        disposition,
        transcript,
    ))
}

pub(crate) fn validate_control(
    control: &OwnedFd,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    validate_common_control(control)?;
    let local = rustix::net::getsockname(control)
        .map_err(|source| io_error("inspect control local address", source.into()))?;
    let remote = rustix::net::getpeername(control)
        .map_err(|source| io_error("inspect control remote address", source.into()))?;
    if !is_unnamed_unix_address(&local) || !remote.as_ref().is_some_and(is_unnamed_unix_address) {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control is not an unnamed connected Unix socket pair",
        ));
    }
    Ok(())
}

pub(crate) fn validate_accepted_control(
    control: &OwnedFd,
    expected_service_address: &SocketAddrAny,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    validate_common_control(control)?;
    let local = rustix::net::getsockname(control)
        .map_err(|source| io_error("inspect accepted control local address", source.into()))?;
    let remote = rustix::net::getpeername(control)
        .map_err(|source| io_error("inspect accepted control remote address", source.into()))?;
    if local != *expected_service_address || !remote.as_ref().is_some_and(is_unnamed_unix_address) {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control is not the expected canonical filesystem-path service endpoint connected to an unnamed client",
        ));
    }
    Ok(())
}

fn validate_common_control(control: &OwnedFd) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    let descriptor_flags = rustix::io::fcntl_getfd(control)
        .map_err(|source| io_error("inspect control descriptor flags", source.into()))?;
    let status = rustix::fs::fcntl_getfl(control)
        .map_err(|source| io_error("inspect control status flags", source.into()))?;
    if descriptor_flags != rustix::io::FdFlags::CLOEXEC || status != OFlags::RDWR | OFlags::NONBLOCK
    {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control is not exact nonblocking close-on-exec read/write custody",
        ));
    }
    if rustix::net::sockopt::socket_domain(control)
        .map_err(|source| io_error("inspect control socket domain", source.into()))?
        != AddressFamily::UNIX
        || rustix::net::sockopt::socket_type(control)
            .map_err(|source| io_error("inspect control socket type", source.into()))?
            != SocketType::SEQPACKET
        || rustix::net::sockopt::socket_acceptconn(control)
            .map_err(|source| io_error("inspect control listener state", source.into()))?
    {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control is not a connected Unix SOCK_SEQPACKET endpoint",
        ));
    }
    Ok(())
}

fn is_unnamed_unix_address(address: &SocketAddrAny) -> bool {
    *address == SocketAddrAny::from(SocketAddrUnix::new_unnamed())
}

pub(crate) fn canonical_filesystem_unix_address(path: &Path) -> Option<SocketAddrAny> {
    let path_bytes = path.as_os_str().as_bytes();
    if path_bytes.is_empty()
        || path_bytes.len() > LINUX_MAX_CANONICAL_FILESYSTEM_SOCKET_PATH_BYTES
        || path_bytes.first() != Some(&b'/')
        || path_bytes.last() == Some(&b'/')
    {
        return None;
    }
    let mut components = path_bytes.split(|byte| *byte == b'/');
    if components.next() != Some(&[][..])
        || !components
            .all(|component| !component.is_empty() && component != b"." && component != b"..")
    {
        return None;
    }
    SocketAddrUnix::new(path).ok().map(SocketAddrAny::from)
}

pub(crate) fn caller_identity(
    control: &OwnedFd,
) -> Result<WorkerV3VerificationCallerV1, WorkerV3VerificationServiceErrorV1> {
    let credentials = rustix::net::sockopt::socket_peercred(control)
        .map_err(|source| io_error("inspect control SO_PEERCRED", source.into()))?;
    let pid = u32::try_from(credentials.pid.as_raw_nonzero().get()).map_err(|_| {
        WorkerV3VerificationServiceErrorV1::InvalidControl(
            "SO_PEERCRED PID does not fit a positive u32",
        )
    })?;
    Ok(WorkerV3VerificationCallerV1 {
        pid,
        uid: credentials.uid.as_raw(),
        gid: credentials.gid.as_raw(),
    })
}

/// Enables kernel-stamped credentials before a peer can submit a request packet.
///
/// Call this on the listener before `listen`/`accept`, or on a private socket-pair receiver before
/// the sender endpoint is exposed. Enabling `SO_PASSCRED` only after a peer may already have queued
/// a packet is too late and will fail closed during [`serve_worker_v3_verification_session_v1`].
/// This transport preparation grants no verification authority.
pub fn prepare_worker_v3_verification_receiver_v1(
    control: &impl AsFd,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    rustix::net::sockopt::set_socket_passcred(control, true)
        .map_err(|source| io_error("enable control SO_PASSCRED", source.into()))?;
    require_passcred(control)
}

pub(crate) fn require_passcred(
    control: &impl AsFd,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    if !rustix::net::sockopt::socket_passcred(control)
        .map_err(|source| io_error("confirm control SO_PASSCRED", source.into()))?
    {
        return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
            "control did not retain SO_PASSCRED",
        ));
    }
    Ok(())
}

pub(crate) fn receive_request(
    control: &OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    deadline: Instant,
) -> Result<(Vec<u8>, [OwnedFd; 2]), WorkerV3VerificationServiceErrorV1> {
    receive_request_sized(
        control,
        caller,
        deadline,
        MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1,
    )
}

pub(crate) fn receive_request_sized(
    control: &OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    deadline: Instant,
    maximum: usize,
) -> Result<(Vec<u8>, [OwnedFd; 2]), WorkerV3VerificationServiceErrorV1> {
    loop {
        wait_for(control, PollFlags::IN, deadline)?;
        let mut payload = vec![0_u8; maximum];
        let received = {
            let mut vectors = [IoSliceMut::new(&mut payload)];
            let mut space = AlignedAncillaryStorageV1(
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2), ScmCredentials(1))],
            );
            let mut ancillary = RecvAncillaryBuffer::new(&mut space.0);
            match recvmsg(
                control,
                &mut vectors,
                &mut ancillary,
                RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC,
            ) {
                Ok(received) => {
                    let mut descriptors = Vec::with_capacity(2);
                    let mut invalid_ancillary = false;
                    let mut rights_messages = 0_usize;
                    let mut credentials = None;
                    for message in ancillary.drain() {
                        match message {
                            RecvAncillaryMessage::ScmRights(received) => {
                                rights_messages += 1;
                                descriptors.extend(received);
                            }
                            RecvAncillaryMessage::ScmCredentials(received) => {
                                if credentials.replace(received).is_some() {
                                    invalid_ancillary = true;
                                }
                            }
                            _ => invalid_ancillary = true,
                        }
                    }
                    if received.bytes == 0 && credentials.is_none() && descriptors.is_empty() {
                        return Err(WorkerV3VerificationServiceErrorV1::PeerClosed);
                    }
                    let matching_credentials = credentials.is_some_and(|credentials| {
                        u32::try_from(credentials.pid.as_raw_pid()).ok() == Some(caller.pid)
                            && credentials.uid.as_raw() == caller.uid
                            && credentials.gid.as_raw() == caller.gid
                    });
                    if received
                        .flags
                        .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
                        || invalid_ancillary
                        || rights_messages != 1
                        || descriptors.len() != WORKER_V3_VERIFICATION_FD_PAYLOADS_V1
                        || !matching_credentials
                        || received.bytes == 0
                    {
                        return Err(WorkerV3VerificationServiceErrorV1::MalformedTransfer);
                    }
                    payload.truncate(received.bytes);
                    let finalized_hsaco = descriptors.pop().expect("descriptor count checked");
                    let load_envelope = descriptors.pop().expect("descriptor count checked");
                    Ok((payload, [load_envelope, finalized_hsaco]))
                }
                Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
                Err(source) => {
                    return Err(io_error("receive request and SCM_RIGHTS", source.into()));
                }
            }
        };
        return received;
    }
}

pub(crate) fn require_peer_write_eof(
    control: &OwnedFd,
    deadline: Instant,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    loop {
        let events = wait_for_eof(control, deadline)?;
        let mut trailing = [0_u8; 1];
        let result = {
            let mut vectors = [IoSliceMut::new(&mut trailing)];
            let mut space = AlignedAncillaryStorageV1(
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1), ScmCredentials(1))],
            );
            let mut ancillary = RecvAncillaryBuffer::new(&mut space.0);
            match recvmsg(
                control,
                &mut vectors,
                &mut ancillary,
                RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC,
            ) {
                Ok(received) => {
                    let mut has_ancillary = false;
                    for message in ancillary.drain() {
                        has_ancillary = true;
                        drop(message);
                    }
                    if received.bytes == 0
                        && !has_ancillary
                        && events.intersects(PollFlags::RDHUP | PollFlags::HUP)
                        && !received
                            .flags
                            .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
                    {
                        Ok(())
                    } else {
                        Err(WorkerV3VerificationServiceErrorV1::TrailingTransfer)
                    }
                }
                Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
                Err(source) => {
                    return Err(io_error("require request write-half EOF", source.into()));
                }
            }
        };
        return result;
    }
}

fn wait_for_eof(
    control: &OwnedFd,
    deadline: Instant,
) -> Result<PollFlags, WorkerV3VerificationServiceErrorV1> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(WorkerV3VerificationServiceErrorV1::Timeout);
        }
        let timeout = Timespec {
            tv_sec: i64::try_from(remaining.as_secs()).unwrap_or(i64::MAX),
            tv_nsec: i64::from(remaining.subsec_nanos()),
        };
        let mut descriptors = [PollFd::new(
            control,
            PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP,
        )];
        match poll(&mut descriptors, Some(&timeout)) {
            Ok(0) => return Err(WorkerV3VerificationServiceErrorV1::Timeout),
            Ok(_) => {
                let events = descriptors[0].revents();
                if events.contains(PollFlags::NVAL) {
                    return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
                        "control descriptor became invalid while requiring EOF",
                    ));
                }
                if events.intersects(PollFlags::IN | PollFlags::RDHUP | PollFlags::HUP) {
                    return Ok(events);
                }
                if events.contains(PollFlags::ERR) {
                    return Err(WorkerV3VerificationServiceErrorV1::PeerClosed);
                }
            }
            Err(rustix::io::Errno::INTR) => continue,
            Err(source) => return Err(io_error("poll for request write-half EOF", source.into())),
        }
    }
}

pub(crate) fn object_key(descriptor: &OwnedFd) -> io::Result<(u64, u64)> {
    let stat = rustix::fs::fstat(descriptor).map_err(io::Error::from)?;
    Ok((stat.st_dev, stat.st_ino))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PayloadSourceSnapshotV1 {
    device: u64,
    inode: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    links: u64,
    byte_len: u64,
    status: OFlags,
    seals: SealFlags,
    filesystem: u64,
}

fn source_snapshot(
    source: &OwnedFd,
    caller: WorkerV3VerificationCallerV1,
    descriptor: WorkerV3VerificationFdPayloadDescriptorV1,
) -> Result<PayloadSourceSnapshotV1, WorkerV3VerificationRejectionReasonV1> {
    let kind = descriptor.kind();
    let descriptor_flags = rustix::io::fcntl_getfd(source)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind))?;
    let status = rustix::fs::fcntl_getfl(source)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind))?;
    let seals = rustix::fs::fcntl_get_seals(source)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind))?;
    let stat = rustix::fs::fstat(source)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind))?;
    let filesystem = rustix::fs::fstatfs(source)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind))?
        .f_type as u64;
    let byte_len = u64::try_from(stat.st_size)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadLengthMismatch(kind))?;
    let accepted_seals = seals == REQUIRED_IMMUTABLE_SEALS_V1
        || seals == REQUIRED_IMMUTABLE_SEALS_V1 | SealFlags::FUTURE_WRITE;
    if descriptor_flags != rustix::io::FdFlags::CLOEXEC
        || status & OFlags::ACCMODE != OFlags::RDONLY
        || status.intersects(OFlags::APPEND | OFlags::ASYNC | OFlags::DIRECT | OFlags::PATH)
        || !accepted_seals
        || FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 0
        || stat.st_uid != caller.uid
        || stat.st_gid != caller.gid
        || filesystem != TMPFS_MAGIC_V1
    {
        return Err(WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind));
    }
    if byte_len != descriptor.byte_len() {
        return Err(WorkerV3VerificationRejectionReasonV1::PayloadLengthMismatch(kind));
    }
    Ok(PayloadSourceSnapshotV1 {
        device: stat.st_dev,
        inode: stat.st_ino,
        mode: stat.st_mode,
        uid: stat.st_uid,
        gid: stat.st_gid,
        links: stat.st_nlink,
        byte_len,
        status,
        seals,
        filesystem,
    })
}

pub(crate) fn capture_payload(
    source: OwnedFd,
    descriptor: WorkerV3VerificationFdPayloadDescriptorV1,
    caller: WorkerV3VerificationCallerV1,
) -> Result<RetainedWorkerV3VerificationPayloadV1, WorkerV3VerificationRejectionReasonV1> {
    let kind = descriptor.kind();
    let before = source_snapshot(&source, caller, descriptor)?;
    let destination = rustix::fs::memfd_create(
        match kind {
            WorkerV3VerificationFdPayloadKindV1::LoadEnvelopeV2 => {
                "fe2o3-worker-v3-load-envelope-v2"
            }
            WorkerV3VerificationFdPayloadKindV1::FinalizedHsaco => {
                "fe2o3-worker-v3-finalized-hsaco"
            }
            _ => "fe2o3-worker-v3-payload",
        },
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let mut writer = File::from(destination);
    rustix::fs::fchmod(&writer, Mode::RUSR)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; PAYLOAD_COPY_CHUNK_BYTES_V1];
    let mut offset = 0_u64;
    while offset < descriptor.byte_len() {
        let count = usize::try_from(
            (descriptor.byte_len() - offset).min(PAYLOAD_COPY_CHUNK_BYTES_V1 as u64),
        )
        .expect("bounded payload chunk fits usize");
        let read = rustix::io::pread(&source, &mut buffer[..count], offset)
            .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadBoundaryMismatch(kind))?;
        if read != count {
            return Err(WorkerV3VerificationRejectionReasonV1::PayloadBoundaryMismatch(kind));
        }
        digest.update(&buffer[..read]);
        writer
            .write_all(&buffer[..read])
            .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
        offset = offset
            .checked_add(read as u64)
            .ok_or(WorkerV3VerificationRejectionReasonV1::PayloadBoundaryMismatch(kind))?;
    }
    let mut trailing = [0_u8; 1];
    if rustix::io::pread(&source, &mut trailing, descriptor.byte_len())
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadBoundaryMismatch(kind))?
        != 0
    {
        return Err(WorkerV3VerificationRejectionReasonV1::PayloadBoundaryMismatch(kind));
    }
    let observed_sha256: [u8; 32] = digest.finalize().into();
    if observed_sha256 != *descriptor.sha256() {
        return Err(WorkerV3VerificationRejectionReasonV1::PayloadDigestMismatch(kind));
    }
    if source_snapshot(&source, caller, descriptor)? != before {
        return Err(WorkerV3VerificationRejectionReasonV1::InvalidPayloadDescriptor(kind));
    }
    writer
        .flush()
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    rustix::fs::fcntl_add_seals(&writer, REQUIRED_IMMUTABLE_SEALS_V1)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let writer_stat = rustix::fs::fstat(&writer)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let path = format!("/proc/self/fd/{}", writer.as_raw_fd());
    let retained = rustix::fs::open(&path, OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty())
        .map(File::from)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let retained_stat = rustix::fs::fstat(&retained)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let retained_flags = rustix::io::fcntl_getfd(&retained)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let retained_status = rustix::fs::fcntl_getfl(&retained)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let retained_seals = rustix::fs::fcntl_get_seals(&retained)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?;
    let retained_filesystem = rustix::fs::fstatfs(&retained)
        .map_err(|_| WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(kind))?
        .f_type as u64;
    if retained_stat.st_dev != writer_stat.st_dev
        || retained_stat.st_ino != writer_stat.st_ino
        || FileType::from_raw_mode(retained_stat.st_mode) != FileType::RegularFile
        || retained_stat.st_mode & 0o7777 != Mode::RUSR.bits()
        || retained_stat.st_nlink != 0
        || u64::try_from(retained_stat.st_size).ok() != Some(descriptor.byte_len())
        || retained_flags != rustix::io::FdFlags::CLOEXEC
        || retained_status & OFlags::ACCMODE != OFlags::RDONLY
        || retained_status
            .intersects(OFlags::APPEND | OFlags::ASYNC | OFlags::DIRECT | OFlags::PATH)
        || retained_seals != REQUIRED_IMMUTABLE_SEALS_V1
        || retained_filesystem != TMPFS_MAGIC_V1
    {
        return Err(WorkerV3VerificationRejectionReasonV1::PayloadCustodyFailed(
            kind,
        ));
    }
    drop(writer);
    drop(source);
    Ok(RetainedWorkerV3VerificationPayloadV1 {
        file: retained,
        kind,
        byte_len: descriptor.byte_len(),
        sha256: observed_sha256,
    })
}

pub(crate) fn send_response(
    control: &OwnedFd,
    bytes: &[u8],
    deadline: Instant,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    loop {
        wait_for(control, PollFlags::OUT, deadline)?;
        match rustix::net::send(control, bytes, SendFlags::DONTWAIT | SendFlags::NOSIGNAL) {
            Ok(sent) if sent == bytes.len() => return Ok(()),
            Ok(_) => return Err(WorkerV3VerificationServiceErrorV1::PartialSend),
            Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
            Err(source) => return Err(io_error("send framing response", source.into())),
        }
    }
}

pub(crate) fn wait_for(
    control: &OwnedFd,
    wanted: PollFlags,
    deadline: Instant,
) -> Result<(), WorkerV3VerificationServiceErrorV1> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(WorkerV3VerificationServiceErrorV1::Timeout);
        }
        let timeout = Timespec {
            tv_sec: i64::try_from(remaining.as_secs()).unwrap_or(i64::MAX),
            tv_nsec: i64::from(remaining.subsec_nanos()),
        };
        let mut descriptors = [PollFd::new(
            control,
            wanted | PollFlags::ERR | PollFlags::HUP,
        )];
        match poll(&mut descriptors, Some(&timeout)) {
            Ok(0) => return Err(WorkerV3VerificationServiceErrorV1::Timeout),
            Ok(_) => {
                let events = descriptors[0].revents();
                if events.contains(PollFlags::NVAL) {
                    return Err(WorkerV3VerificationServiceErrorV1::InvalidControl(
                        "control descriptor became invalid",
                    ));
                }
                if events.intersects(PollFlags::ERR | PollFlags::HUP) && !events.intersects(wanted)
                {
                    return Err(WorkerV3VerificationServiceErrorV1::PeerClosed);
                }
                if events.intersects(wanted) {
                    return Ok(());
                }
            }
            Err(rustix::io::Errno::INTR) => continue,
            Err(source) => return Err(io_error("poll service control", source.into())),
        }
    }
}

fn io_error(operation: &'static str, source: io::Error) -> WorkerV3VerificationServiceErrorV1 {
    WorkerV3VerificationServiceErrorV1::Io { operation, source }
}

/// Stable terminal transport/session error.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationServiceErrorV1 {
    /// The caller supplied a zero timeout.
    InvalidTimeout,
    /// The absolute monotonic deadline cannot be represented.
    DeadlineOverflow,
    /// The service control endpoint has an invalid shape or identity.
    InvalidControl(&'static str),
    /// The one-packet session deadline expired.
    Timeout,
    /// The peer closed before the exchange completed.
    PeerClosed,
    /// The packet was truncated or did not carry exactly two SCM_RIGHTS descriptors.
    MalformedTransfer,
    /// The peer sent another packet or ancillary object instead of exact write-half EOF.
    TrailingTransfer,
    /// The request packet was not one exact canonical protocol frame.
    CanonicalRequest(WorkerV3VerificationProtocolErrorV1),
    /// A domain-separated transcript identity unexpectedly used the zero sentinel.
    TranscriptIdentity,
    /// A seqpacket response was not transmitted atomically in full.
    PartialSend,
    /// A bounded Linux descriptor or socket operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Kernel or standard-library failure.
        source: io::Error,
    },
}

impl fmt::Display for WorkerV3VerificationServiceErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeout => formatter.write_str("Worker V3 verification timeout is zero"),
            Self::DeadlineOverflow => {
                formatter.write_str("Worker V3 verification deadline overflowed")
            }
            Self::InvalidControl(reason) => write!(formatter, "invalid service control: {reason}"),
            Self::Timeout => formatter.write_str("Worker V3 verification session timed out"),
            Self::PeerClosed => formatter.write_str("Worker V3 verification peer closed"),
            Self::MalformedTransfer => formatter.write_str(
                "Worker V3 verification transfer is not one packet with exactly two descriptors",
            ),
            Self::TrailingTransfer => formatter.write_str(
                "Worker V3 verification transfer contains another packet before write-half EOF",
            ),
            Self::CanonicalRequest(source) => {
                write!(
                    formatter,
                    "invalid canonical Worker V3 verification request: {source}"
                )
            }
            Self::TranscriptIdentity => {
                formatter.write_str("Worker V3 verification transcript identity is zero")
            }
            Self::PartialSend => {
                formatter.write_str("Worker V3 verification response send was partial")
            }
            Self::Io { operation, source } => write!(formatter, "{operation}: {source}"),
        }
    }
}

impl Error for WorkerV3VerificationServiceErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CanonicalRequest(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_path_validator_rejects_empty_relative_noncanonical_and_overlong_names() {
        assert!(
            canonical_filesystem_unix_address(Path::new("/run/fe2o3/worker-v3-verifier.sock"))
                .is_some()
        );

        for path in [
            "",
            "relative.sock",
            "/run//verifier.sock",
            "/run/./verifier.sock",
            "/run/../verifier.sock",
            "/run/verifier.sock/",
        ] {
            assert!(
                canonical_filesystem_unix_address(Path::new(path)).is_none(),
                "{path}"
            );
        }

        let overlong = format!("/{}", "x".repeat(107));
        assert!(canonical_filesystem_unix_address(Path::new(&overlong)).is_none());

        let nul = Path::new(std::ffi::OsStr::from_bytes(b"/run/fe2o3/\0verifier.sock"));
        assert!(canonical_filesystem_unix_address(nul).is_none());
    }
}
