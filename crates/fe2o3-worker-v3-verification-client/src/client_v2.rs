use std::error::Error;
use std::fmt;
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::time::{Duration, Instant};

use fe2o3_compiler_execution_client::CompilerExecutionCurrentRecordChallengeV1;
use fe2o3_runtime_protocol::{
    COMPILER_EXECUTION_CURRENT_RECORD_ATTESTATION_BYTES_V3,
    COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3,
    CompilerExecutionCurrentRecordVerificationErrorV3,
};
use fe2o3_worker_v3_verification_protocol::{
    MAX_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2, MIN_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2,
    WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2, WorkerV3VerificationChallengeDispositionV2,
    WorkerV3VerificationChallengeFrameV2, WorkerV3VerificationChallengeReservationV2,
    WorkerV3VerificationCurrentRecordFrameV2, WorkerV3VerificationProtocolErrorV1,
    WorkerV3VerificationProtocolErrorV2, WorkerV3VerificationRequestV1,
    WorkerV3VerificationTerminalFrameV2,
};
use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
    SendAncillaryBuffer, SendAncillaryMessage, SendFlags, Shutdown, SocketAddrAny, SocketAddrUnix,
    SocketType,
};

use crate::{WorkerV3VerificationClientErrorV1, WorkerV3VerificationPayloadSnapshotsV1};

const LINUX_SA_FAMILY_BYTES: u32 = 2;
const LINUX_MAX_CANONICAL_FILESYSTEM_SOCKET_PATH_BYTES: usize = 107;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PeerCredentialsV2 {
    pid: u32,
    uid: u32,
    gid: u32,
}

/// One owned V2 connection whose absolute deadline covers every protocol phase.
///
/// ```compile_fail
/// use fe2o3_worker_v3_verification_client::WorkerV3VerificationClientV2;
/// fn duplicate(value: WorkerV3VerificationClientV2) { let _again = value.clone(); }
/// ```
pub struct WorkerV3VerificationClientV2 {
    peer: OwnedFd,
    deadline: Instant,
    expected_response_credentials: Option<PeerCredentialsV2>,
}

impl fmt::Debug for WorkerV3VerificationClientV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationClientV2")
            .field("deadline", &self.deadline)
            .field("peer_authority", &"none")
            .finish_non_exhaustive()
    }
}

impl WorkerV3VerificationClientV2 {
    /// Admits one connected unnamed Unix `SOCK_SEQPACKET` peer for one bounded timeout.
    pub fn admit(
        peer: OwnedFd,
        timeout: Duration,
    ) -> Result<Self, WorkerV3VerificationClientErrorV2> {
        if timeout.is_zero() {
            return Err(WorkerV3VerificationClientErrorV2::InvalidTimeout);
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(WorkerV3VerificationClientErrorV2::DeadlineOverflow)?;
        Self::admit_until(peer, deadline)
    }

    /// Admits one connected unnamed Unix `SOCK_SEQPACKET` peer until an exact deadline.
    ///
    /// The caller-supplied monotonic deadline is retained unchanged across every V2 phase.
    pub fn admit_until(
        peer: OwnedFd,
        deadline: Instant,
    ) -> Result<Self, WorkerV3VerificationClientErrorV2> {
        require_deadline(deadline)?;
        set_close_on_exec(&peer)?;
        validate_peer(&peer)?;
        Ok(Self {
            peer,
            deadline,
            expected_response_credentials: None,
        })
    }

    /// Admits one already-connected filesystem-path service endpoint for one bounded timeout.
    ///
    /// This function never discovers or connects to a pathname. The supplied endpoint must have
    /// an unnamed local address and a peer address exactly matching the caller-supplied canonical
    /// absolute `expected_service_path`. Admission also enables `SO_PASSCRED`, snapshots the peer's
    /// `SO_PEERCRED`, and requires every V2 response packet to carry exactly matching kernel-stamped
    /// `SCM_CREDENTIALS`.
    ///
    /// Failure retains ownership of the supplied endpoint.
    pub fn admit_connected_path(
        peer: OwnedFd,
        expected_service_path: &Path,
        timeout: Duration,
    ) -> Result<Self, WorkerV3VerificationClientAdmissionFailureV2> {
        let deadline = if timeout.is_zero() {
            return Err(WorkerV3VerificationClientAdmissionFailureV2 {
                peer,
                source: WorkerV3VerificationClientErrorV2::InvalidTimeout,
            });
        } else {
            match Instant::now().checked_add(timeout) {
                Some(deadline) => deadline,
                None => {
                    return Err(WorkerV3VerificationClientAdmissionFailureV2 {
                        peer,
                        source: WorkerV3VerificationClientErrorV2::DeadlineOverflow,
                    });
                }
            }
        };
        Self::admit_connected_path_until(peer, expected_service_path, deadline)
    }

    /// Admits one already-connected filesystem-path service endpoint until an exact deadline.
    ///
    /// The caller-supplied monotonic deadline and the connection-time peer credentials are retained
    /// unchanged across every V2 phase. This API accepts only the asymmetric client side of an
    /// accepted pathname connection; the existing unnamed admission APIs remain strict.
    ///
    /// Failure retains ownership of the supplied endpoint.
    pub fn admit_connected_path_until(
        peer: OwnedFd,
        expected_service_path: &Path,
        deadline: Instant,
    ) -> Result<Self, WorkerV3VerificationClientAdmissionFailureV2> {
        let admitted = (|| {
            require_deadline(deadline)?;
            let expected_service_address = canonical_filesystem_unix_address(expected_service_path)
                .ok_or(WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer)?;
            set_close_on_exec(&peer)?;
            validate_connected_path_peer(&peer, &expected_service_address)?;
            enable_passcred(&peer)?;
            validate_connected_path_peer(&peer, &expected_service_address)?;
            let expected_response_credentials = peer_credentials(&peer)?;
            require_deadline(deadline)?;
            Ok(expected_response_credentials)
        })();
        match admitted {
            Ok(expected_response_credentials) => Ok(Self {
                peer,
                deadline,
                expected_response_credentials: Some(expected_response_credentials),
            }),
            Err(source) => Err(WorkerV3VerificationClientAdmissionFailureV2 { peer, source }),
        }
    }

    /// Returns the exact caller-supplied monotonic deadline retained by this session.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Sends the canonical Begin request and exactly two immutable payload descriptors.
    ///
    /// A successful result contains a move-only service challenge and a separate pending session.
    /// A rejected Begin consumes the connection and carries only authority-free framing evidence.
    pub fn begin(
        self,
        request: WorkerV3VerificationRequestV1,
        snapshots: WorkerV3VerificationPayloadSnapshotsV1,
    ) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationClientErrorV2> {
        self.begin_with_readiness(request, snapshots, None)
    }

    pub(crate) fn receive_readiness(&self) -> Result<Vec<u8>, WorkerV3VerificationClientErrorV2> {
        receive_packet(
            &self.peer,
            64,
            64,
            self.deadline,
            self.expected_response_credentials,
        )
    }

    pub(crate) fn require_readiness_quiet(&self) -> Result<(), WorkerV3VerificationClientErrorV2> {
        loop {
            require_deadline(self.deadline)?;
            let mut fds = [PollFd::new(
                &self.peer,
                PollFlags::IN | PollFlags::RDHUP | PollFlags::ERR | PollFlags::HUP,
            )];
            match poll(
                &mut fds,
                Some(&Timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }),
            ) {
                Ok(0) => return Ok(()),
                Ok(_) => return Err(WorkerV3VerificationClientErrorV2::TrailingTransfer),
                Err(rustix::io::Errno::INTR) => {}
                Err(e) => return Err(WorkerV3VerificationClientErrorV2::Poll(e.into())),
            }
        }
    }

    pub(crate) fn begin_with_readiness(
        self,
        request: WorkerV3VerificationRequestV1,
        snapshots: WorkerV3VerificationPayloadSnapshotsV1,
        ready: Option<fe2o3_worker_v3_verification_protocol::WorkerV3VerificationReadyFrameV3>,
    ) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationClientErrorV2> {
        snapshots
            .revalidate(&request)
            .map_err(WorkerV3VerificationClientErrorV2::Snapshot)?;
        let bytes: std::borrow::Cow<'_, [u8]> = match ready {
            Some(ready) => std::borrow::Cow::Owned(ready.encode_begin(&request)),
            None => std::borrow::Cow::Borrowed(request.encode_canonical()),
        };
        send_begin(&self.peer, &bytes, snapshots.borrowed_fds(), self.deadline)?;
        snapshots
            .revalidate(&request)
            .map_err(WorkerV3VerificationClientErrorV2::Snapshot)?;
        let bytes = receive_packet(
            &self.peer,
            WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2,
            WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2,
            self.deadline,
            self.expected_response_credentials,
        )?;
        let frame = WorkerV3VerificationChallengeFrameV2::decode_canonical(&bytes)?;
        if !frame.matches_request(&request) {
            return Err(WorkerV3VerificationClientErrorV2::SessionMismatch);
        }
        match frame.disposition() {
            WorkerV3VerificationChallengeDispositionV2::Reserved => {
                let reservation = frame
                    .into_reservation()
                    .ok_or(WorkerV3VerificationClientErrorV2::MissingReservation)?;
                let expected_challenge = *reservation.challenge_bytes();
                let expected_reservation_identity = *reservation.reservation_identity();
                Ok(WorkerV3VerificationBeginOutcomeV2::Reserved(
                    WorkerV3VerificationReservedBeginV2 {
                        challenge: WorkerV3VerificationCurrentRecordChallengeV2 { reservation },
                        pending: PendingWorkerV3VerificationClientV2 {
                            peer: self.peer,
                            deadline: self.deadline,
                            request,
                            expected_challenge,
                            expected_reservation_identity,
                            expected_response_credentials: self.expected_response_credentials,
                        },
                    },
                ))
            }
            WorkerV3VerificationChallengeDispositionV2::Rejected => {
                require_peer_eof(&self.peer, self.deadline)?;
                Ok(WorkerV3VerificationBeginOutcomeV2::Rejected(
                    RejectedWorkerV3VerificationBeginV2 { request, frame },
                ))
            }
            _ => Err(WorkerV3VerificationClientErrorV2::UnexpectedDisposition),
        }
    }

    /// Reports that socket admission does not authenticate the service peer.
    pub const fn authenticates_peer(&self) -> bool {
        false
    }
}

/// Move-only challenge released by the service for one pending V2 session.
///
/// ```compile_fail
/// use fe2o3_worker_v3_verification_client::WorkerV3VerificationCurrentRecordChallengeV2;
/// fn duplicate(value: WorkerV3VerificationCurrentRecordChallengeV2) {
///     let _again = value.clone();
/// }
/// ```
pub struct WorkerV3VerificationCurrentRecordChallengeV2 {
    reservation: WorkerV3VerificationChallengeReservationV2,
}

impl WorkerV3VerificationCurrentRecordChallengeV2 {
    /// Borrows the exact service-issued bytes for constructing a compiler-current request.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.reservation.challenge_bytes()
    }

    /// Borrows the opaque reservation identity bound into every later phase.
    pub const fn reservation_identity(&self) -> &[u8; 32] {
        self.reservation.reservation_identity()
    }

    /// Consumes the move-only challenge into inert bytes for another reviewed client boundary.
    pub fn into_bytes(self) -> [u8; 32] {
        self.reservation.into_bytes().0
    }

    /// Consumes this transport token into the compiler client's move-only caller challenge.
    ///
    /// The conversion preserves the exact service-issued bytes. The still-pending V2 transport
    /// privately retains the same bytes solely to correlate the later current-record frame.
    pub fn into_compiler_execution_challenge(
        self,
    ) -> Result<
        CompilerExecutionCurrentRecordChallengeV1,
        CompilerExecutionCurrentRecordVerificationErrorV3,
    > {
        CompilerExecutionCurrentRecordChallengeV1::from_bytes(self.into_bytes())
    }

    /// Reports that receipt of a challenge grants no authority or durability guarantee.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for WorkerV3VerificationCurrentRecordChallengeV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationCurrentRecordChallengeV2")
            .field(
                "reservation_identity",
                &self.reservation.reservation_identity(),
            )
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

/// Successful first phase containing separable move-only challenge and pending transport custody.
pub struct WorkerV3VerificationReservedBeginV2 {
    challenge: WorkerV3VerificationCurrentRecordChallengeV2,
    pending: PendingWorkerV3VerificationClientV2,
}

impl WorkerV3VerificationReservedBeginV2 {
    /// Separates the challenge for compiler-current acquisition from the still-open V2 session.
    pub fn into_parts(
        self,
    ) -> (
        WorkerV3VerificationCurrentRecordChallengeV2,
        PendingWorkerV3VerificationClientV2,
    ) {
        (self.challenge, self.pending)
    }

    /// Reports that the first phase grants no authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for WorkerV3VerificationReservedBeginV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationReservedBeginV2")
            .field("challenge", &self.challenge)
            .field("pending", &self.pending)
            .field("authority", &"none")
            .finish()
    }
}

/// Terminal first-phase rejection.
pub struct RejectedWorkerV3VerificationBeginV2 {
    request: WorkerV3VerificationRequestV1,
    frame: WorkerV3VerificationChallengeFrameV2,
}

impl RejectedWorkerV3VerificationBeginV2 {
    /// Returns the exact rejected Begin request.
    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    /// Returns the correlated generic rejection frame.
    pub const fn frame(&self) -> &WorkerV3VerificationChallengeFrameV2 {
        &self.frame
    }

    /// Reports that a rejection grants no authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl fmt::Debug for RejectedWorkerV3VerificationBeginV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedWorkerV3VerificationBeginV2")
            .field("request", &self.request.identity())
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

/// Result of the V2 Begin phase.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationBeginOutcomeV2 {
    /// The service reserved a challenge and retained the exact Begin payloads.
    Reserved(WorkerV3VerificationReservedBeginV2),
    /// The service rejected the decoded Begin request.
    Rejected(RejectedWorkerV3VerificationBeginV2),
}

impl WorkerV3VerificationBeginOutcomeV2 {
    /// Reports that neither outcome grants authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Move-only pending client session after the challenge has been released.
///
/// ```compile_fail
/// use fe2o3_worker_v3_verification_client::PendingWorkerV3VerificationClientV2;
/// fn duplicate(value: PendingWorkerV3VerificationClientV2) { let _again = value.clone(); }
/// ```
pub struct PendingWorkerV3VerificationClientV2 {
    peer: OwnedFd,
    deadline: Instant,
    request: WorkerV3VerificationRequestV1,
    expected_challenge: [u8; 32],
    expected_reservation_identity: [u8; 32],
    expected_response_credentials: Option<PeerCredentialsV2>,
}

impl fmt::Debug for PendingWorkerV3VerificationClientV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingWorkerV3VerificationClientV2")
            .field("request", &self.request.identity())
            .field("deadline", &self.deadline)
            .field("reservation_identity", &self.expected_reservation_identity)
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}

impl PendingWorkerV3VerificationClientV2 {
    /// Submits the exact fixed-size compiler-current records and receives the sole terminal frame.
    pub fn submit_current_record(
        self,
        verification: [u8; COMPILER_EXECUTION_CURRENT_RECORD_VERIFICATION_BYTES_V3],
        attestation: [u8; COMPILER_EXECUTION_CURRENT_RECORD_ATTESTATION_BYTES_V3],
    ) -> Result<WorkerV3VerificationTerminalFrameV2, WorkerV3VerificationClientErrorV2> {
        let reservation = WorkerV3VerificationChallengeReservationV2::new(
            self.expected_challenge,
            self.expected_reservation_identity,
        )?;
        let frame = WorkerV3VerificationCurrentRecordFrameV2::new(
            &self.request,
            &reservation,
            &verification,
            &attestation,
        )?;
        send_packet(&self.peer, frame.encode_canonical(), self.deadline)?;
        rustix::net::shutdown(&self.peer, Shutdown::Write)
            .map_err(|source| WorkerV3VerificationClientErrorV2::Shutdown(source.into()))?;
        let bytes = receive_packet(
            &self.peer,
            MIN_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2,
            MAX_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2,
            self.deadline,
            self.expected_response_credentials,
        )?;
        let terminal = WorkerV3VerificationTerminalFrameV2::decode_canonical(&bytes)?;
        if !terminal.matches_session(&self.request, &reservation) {
            return Err(WorkerV3VerificationClientErrorV2::SessionMismatch);
        }
        require_peer_eof(&self.peer, self.deadline)?;
        Ok(terminal)
    }

    /// Returns the exact Begin request retained for response correlation.
    pub const fn request(&self) -> &WorkerV3VerificationRequestV1 {
        &self.request
    }

    /// Returns the exact admission deadline retained across all remaining phases.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Reports that pending transport state grants no authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Failed connected-filesystem-path V2 admission retaining the caller's endpoint.
pub struct WorkerV3VerificationClientAdmissionFailureV2 {
    peer: OwnedFd,
    source: WorkerV3VerificationClientErrorV2,
}

impl WorkerV3VerificationClientAdmissionFailureV2 {
    /// Returns the exact admission error.
    pub const fn source_error(&self) -> &WorkerV3VerificationClientErrorV2 {
        &self.source
    }

    /// Returns ownership of the endpoint rejected before any protocol bytes were sent.
    pub fn into_peer(self) -> OwnedFd {
        self.peer
    }
}

impl fmt::Debug for WorkerV3VerificationClientAdmissionFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerV3VerificationClientAdmissionFailureV2")
            .field("source", &self.source)
            .field("endpoint_retained", &true)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WorkerV3VerificationClientAdmissionFailureV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "V2 connected-path client admission failed with endpoint retained: {}",
            self.source
        )
    }
}

impl Error for WorkerV3VerificationClientAdmissionFailureV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// V2 admission, phase transport, or correlation failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationClientErrorV2 {
    InvalidTimeout,
    DeadlineOverflow,
    Snapshot(WorkerV3VerificationClientErrorV1),
    ProtocolV1(WorkerV3VerificationProtocolErrorV1),
    ProtocolV2(WorkerV3VerificationProtocolErrorV2),
    Descriptor {
        operation: &'static str,
        source: io::Error,
    },
    NotSeqpacket,
    NamedOrNonUnixPeer,
    InvalidConnectedPathPeer,
    ResponseCredentialsMismatch,
    Poll(io::Error),
    Send(io::Error),
    Shutdown(io::Error),
    Receive(io::Error),
    Timeout,
    InvalidPeer,
    PeerFailed,
    PeerClosed,
    PartialSend {
        expected: usize,
        actual: usize,
    },
    PacketTruncated {
        minimum: usize,
        actual: usize,
    },
    PacketOversize {
        maximum: usize,
        actual: usize,
    },
    UnexpectedAncillaryData,
    TrailingTransfer,
    SessionMismatch,
    MissingReservation,
    UnexpectedDisposition,
}

impl fmt::Display for WorkerV3VerificationClientErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeout => formatter.write_str("V2 timeout must be nonzero"),
            Self::DeadlineOverflow => formatter.write_str("V2 absolute deadline overflowed"),
            Self::Snapshot(source) => write!(formatter, "V2 snapshot admission failed: {source}"),
            Self::ProtocolV1(source) => write!(formatter, "V2 Begin framing failed: {source}"),
            Self::ProtocolV2(source) => write!(formatter, "V2 phase framing failed: {source}"),
            Self::Descriptor { operation, source } => {
                write!(
                    formatter,
                    "V2 descriptor operation `{operation}` failed: {source}"
                )
            }
            Self::NotSeqpacket => formatter.write_str("V2 peer is not SOCK_SEQPACKET"),
            Self::NamedOrNonUnixPeer => {
                formatter.write_str("V2 peer is not a connected unnamed Unix socket")
            }
            Self::InvalidConnectedPathPeer => formatter.write_str(
                "V2 peer is not an unnamed client endpoint connected to the expected canonical filesystem pathname",
            ),
            Self::ResponseCredentialsMismatch => formatter.write_str(
                "V2 response credentials differ from the admitted service peer",
            ),
            Self::Poll(source) => write!(formatter, "V2 peer poll failed: {source}"),
            Self::Send(source) => write!(formatter, "V2 packet send failed: {source}"),
            Self::Shutdown(source) => write!(formatter, "V2 write shutdown failed: {source}"),
            Self::Receive(source) => write!(formatter, "V2 packet receive failed: {source}"),
            Self::Timeout => formatter.write_str("V2 absolute deadline expired"),
            Self::InvalidPeer => formatter.write_str("V2 peer descriptor became invalid"),
            Self::PeerFailed => formatter.write_str("V2 peer reported an error"),
            Self::PeerClosed => formatter.write_str("V2 peer closed before the required phase"),
            Self::PartialSend { expected, actual } => {
                write!(
                    formatter,
                    "V2 packet send was partial: expected {expected}, got {actual}"
                )
            }
            Self::PacketTruncated { minimum, actual } => {
                write!(
                    formatter,
                    "V2 packet was shorter than {minimum} bytes: got {actual}"
                )
            }
            Self::PacketOversize { maximum, actual } => {
                write!(
                    formatter,
                    "V2 packet exceeded {maximum} bytes: got {actual}"
                )
            }
            Self::UnexpectedAncillaryData => {
                formatter.write_str("V2 non-Begin packet carried ancillary data")
            }
            Self::TrailingTransfer => formatter.write_str("V2 peer sent a trailing packet"),
            Self::SessionMismatch => formatter.write_str("V2 phase names another session"),
            Self::MissingReservation => {
                formatter.write_str("V2 reserved challenge frame omitted its reservation")
            }
            Self::UnexpectedDisposition => formatter.write_str("V2 phase disposition is invalid"),
        }
    }
}

impl Error for WorkerV3VerificationClientErrorV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Snapshot(source) => Some(source),
            Self::ProtocolV1(source) => Some(source),
            Self::ProtocolV2(source) => Some(source),
            Self::Descriptor { source, .. }
            | Self::Poll(source)
            | Self::Send(source)
            | Self::Shutdown(source)
            | Self::Receive(source) => Some(source),
            _ => None,
        }
    }
}

impl From<WorkerV3VerificationProtocolErrorV2> for WorkerV3VerificationClientErrorV2 {
    fn from(source: WorkerV3VerificationProtocolErrorV2) -> Self {
        Self::ProtocolV2(source)
    }
}

fn validate_peer(peer: &OwnedFd) -> Result<(), WorkerV3VerificationClientErrorV2> {
    validate_common_peer(peer)?;
    let local = rustix::net::getsockname(peer)
        .map_err(|source| descriptor_error("inspect local peer address", source.into()))?;
    let remote = rustix::net::getpeername(peer)
        .map_err(|source| descriptor_error("inspect remote peer address", source.into()))?
        .ok_or(WorkerV3VerificationClientErrorV2::NamedOrNonUnixPeer)?;
    if !is_unnamed_unix_address(&local) || !is_unnamed_unix_address(&remote) {
        return Err(WorkerV3VerificationClientErrorV2::NamedOrNonUnixPeer);
    }
    Ok(())
}

fn validate_connected_path_peer(
    peer: &OwnedFd,
    expected_service_address: &SocketAddrAny,
) -> Result<(), WorkerV3VerificationClientErrorV2> {
    validate_common_peer(peer)?;
    let status = rustix::fs::fcntl_getfl(peer).map_err(|source| {
        descriptor_error("inspect connected-path peer status flags", source.into())
    })?;
    if status != rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NONBLOCK {
        return Err(descriptor_error(
            "require exact connected-path peer status flags",
            io::Error::other(format!("unexpected status flags 0x{:08x}", status.bits())),
        ));
    }
    let local = rustix::net::getsockname(peer).map_err(|source| {
        descriptor_error("inspect connected-path local address", source.into())
    })?;
    let remote = rustix::net::getpeername(peer)
        .map_err(|source| descriptor_error("inspect connected-path peer address", source.into()))?
        .ok_or(WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer)?;
    if !is_unnamed_unix_address(&local) || remote != *expected_service_address {
        return Err(WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer);
    }
    Ok(())
}

fn validate_common_peer(peer: &OwnedFd) -> Result<(), WorkerV3VerificationClientErrorV2> {
    let socket_type = rustix::net::sockopt::socket_type(peer)
        .map_err(|source| descriptor_error("inspect peer socket type", source.into()))?;
    if socket_type != SocketType::SEQPACKET {
        return Err(WorkerV3VerificationClientErrorV2::NotSeqpacket);
    }
    let domain = rustix::net::sockopt::socket_domain(peer)
        .map_err(|source| descriptor_error("inspect peer socket domain", source.into()))?;
    if domain != AddressFamily::UNIX {
        return Err(WorkerV3VerificationClientErrorV2::NamedOrNonUnixPeer);
    }
    Ok(())
}

fn is_unnamed_unix_address(address: &SocketAddrAny) -> bool {
    address.address_family() == AddressFamily::UNIX
        && address.addr_len() == LINUX_SA_FAMILY_BYTES
        && *address == SocketAddrAny::from(SocketAddrUnix::new_unnamed())
}

fn canonical_filesystem_unix_address(path: &Path) -> Option<SocketAddrAny> {
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

fn enable_passcred(peer: &OwnedFd) -> Result<(), WorkerV3VerificationClientErrorV2> {
    rustix::net::sockopt::set_socket_passcred(peer, true).map_err(|source| {
        descriptor_error("enable connected-path peer SO_PASSCRED", source.into())
    })?;
    if !rustix::net::sockopt::socket_passcred(peer).map_err(|source| {
        descriptor_error("confirm connected-path peer SO_PASSCRED", source.into())
    })? {
        return Err(descriptor_error(
            "confirm connected-path peer SO_PASSCRED",
            io::Error::other("connected-path peer did not retain SO_PASSCRED"),
        ));
    }
    Ok(())
}

fn peer_credentials(
    peer: &OwnedFd,
) -> Result<PeerCredentialsV2, WorkerV3VerificationClientErrorV2> {
    let credentials = rustix::net::sockopt::socket_peercred(peer).map_err(|source| {
        descriptor_error("inspect connected-path peer SO_PEERCRED", source.into())
    })?;
    let pid = u32::try_from(credentials.pid.as_raw_nonzero().get()).map_err(|_| {
        descriptor_error(
            "inspect connected-path peer SO_PEERCRED",
            io::Error::other("peer PID does not fit a positive u32"),
        )
    })?;
    Ok(PeerCredentialsV2 {
        pid,
        uid: credentials.uid.as_raw(),
        gid: credentials.gid.as_raw(),
    })
}

fn set_close_on_exec(peer: &OwnedFd) -> Result<(), WorkerV3VerificationClientErrorV2> {
    rustix::io::fcntl_setfd(peer, rustix::io::FdFlags::CLOEXEC)
        .map_err(|source| descriptor_error("set peer close-on-exec", source.into()))?;
    let actual = rustix::io::fcntl_getfd(peer)
        .map_err(|source| descriptor_error("inspect peer descriptor flags", source.into()))?;
    if actual != rustix::io::FdFlags::CLOEXEC {
        return Err(descriptor_error(
            "retain exact peer close-on-exec flags",
            io::Error::other(format!(
                "unexpected descriptor flags 0x{:08x}",
                actual.bits()
            )),
        ));
    }
    Ok(())
}

fn send_begin(
    peer: &OwnedFd,
    bytes: &[u8],
    descriptors: [std::os::fd::BorrowedFd<'_>; 2],
    deadline: Instant,
) -> Result<(), WorkerV3VerificationClientErrorV2> {
    loop {
        wait_for_peer(peer, PollFlags::OUT, deadline)?;
        let mut control_space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2))];
        let mut control = SendAncillaryBuffer::new(&mut control_space);
        if !control.push(SendAncillaryMessage::ScmRights(&descriptors)) {
            return Err(WorkerV3VerificationClientErrorV2::PartialSend {
                expected: bytes.len(),
                actual: 0,
            });
        }
        match rustix::net::sendmsg(
            peer,
            &[IoSlice::new(bytes)],
            &mut control,
            SendFlags::DONTWAIT | SendFlags::NOSIGNAL,
        ) {
            Ok(actual) if actual == bytes.len() => return Ok(()),
            Ok(actual) => {
                return Err(WorkerV3VerificationClientErrorV2::PartialSend {
                    expected: bytes.len(),
                    actual,
                });
            }
            Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => {}
            Err(source) => return Err(WorkerV3VerificationClientErrorV2::Send(source.into())),
        }
    }
}

fn send_packet(
    peer: &OwnedFd,
    bytes: &[u8],
    deadline: Instant,
) -> Result<(), WorkerV3VerificationClientErrorV2> {
    loop {
        wait_for_peer(peer, PollFlags::OUT, deadline)?;
        match rustix::net::send(peer, bytes, SendFlags::DONTWAIT | SendFlags::NOSIGNAL) {
            Ok(actual) if actual == bytes.len() => return Ok(()),
            Ok(actual) => {
                return Err(WorkerV3VerificationClientErrorV2::PartialSend {
                    expected: bytes.len(),
                    actual,
                });
            }
            Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => {}
            Err(source) => return Err(WorkerV3VerificationClientErrorV2::Send(source.into())),
        }
    }
}

fn receive_packet(
    peer: &OwnedFd,
    minimum: usize,
    maximum: usize,
    deadline: Instant,
    expected_credentials: Option<PeerCredentialsV2>,
) -> Result<Vec<u8>, WorkerV3VerificationClientErrorV2> {
    let mut bytes = vec![0_u8; maximum + 1];
    loop {
        wait_for_peer(peer, PollFlags::IN, deadline)?;
        let mut control_space =
            [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1), ScmCredentials(1))];
        let mut control = RecvAncillaryBuffer::new(&mut control_space);
        let received = {
            let mut vectors = [IoSliceMut::new(&mut bytes)];
            match rustix::net::recvmsg(
                peer,
                &mut vectors,
                &mut control,
                RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC | RecvFlags::TRUNC,
            ) {
                Ok(received) => received,
                Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => continue,
                Err(source) => {
                    return Err(WorkerV3VerificationClientErrorV2::Receive(source.into()));
                }
            }
        };
        let mut unexpected_ancillary = false;
        let mut credentials = None;
        for message in control.drain() {
            match message {
                RecvAncillaryMessage::ScmCredentials(received) => {
                    if expected_credentials.is_none() || credentials.replace(received).is_some() {
                        unexpected_ancillary = true;
                    }
                }
                RecvAncillaryMessage::ScmRights(received) => {
                    unexpected_ancillary = true;
                    drop(received);
                }
                _ => unexpected_ancillary = true,
            }
        }
        if received.flags.contains(ReturnFlags::CTRUNC) || unexpected_ancillary {
            return Err(WorkerV3VerificationClientErrorV2::UnexpectedAncillaryData);
        }
        if received.flags.contains(ReturnFlags::TRUNC) || received.bytes > maximum {
            return Err(WorkerV3VerificationClientErrorV2::PacketOversize {
                maximum,
                actual: received.bytes,
            });
        }
        if received.bytes == 0 {
            return Err(WorkerV3VerificationClientErrorV2::PeerClosed);
        }
        if received.bytes < minimum {
            return Err(WorkerV3VerificationClientErrorV2::PacketTruncated {
                minimum,
                actual: received.bytes,
            });
        }
        if expected_credentials.is_some_and(|expected| {
            !credentials.is_some_and(|actual| {
                u32::try_from(actual.pid.as_raw_pid()).ok() == Some(expected.pid)
                    && actual.uid.as_raw() == expected.uid
                    && actual.gid.as_raw() == expected.gid
            })
        }) {
            return Err(WorkerV3VerificationClientErrorV2::ResponseCredentialsMismatch);
        }
        bytes.truncate(received.bytes);
        return Ok(bytes);
    }
}

fn require_peer_eof(
    peer: &OwnedFd,
    deadline: Instant,
) -> Result<(), WorkerV3VerificationClientErrorV2> {
    loop {
        wait_for_peer(peer, PollFlags::IN, deadline)?;
        let mut byte = [0_u8; 1];
        let mut control_space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = RecvAncillaryBuffer::new(&mut control_space);
        let received = {
            let mut vectors = [IoSliceMut::new(&mut byte)];
            match rustix::net::recvmsg(
                peer,
                &mut vectors,
                &mut control,
                RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC | RecvFlags::TRUNC,
            ) {
                Ok(received) => received,
                Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => continue,
                Err(source) => {
                    return Err(WorkerV3VerificationClientErrorV2::Receive(source.into()));
                }
            }
        };
        if received.flags.contains(ReturnFlags::CTRUNC) || control.drain().next().is_some() {
            return Err(WorkerV3VerificationClientErrorV2::UnexpectedAncillaryData);
        }
        if received.bytes == 0 {
            return Ok(());
        }
        return Err(WorkerV3VerificationClientErrorV2::TrailingTransfer);
    }
}

fn wait_for_peer(
    peer: &OwnedFd,
    wanted: PollFlags,
    deadline: Instant,
) -> Result<(), WorkerV3VerificationClientErrorV2> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(WorkerV3VerificationClientErrorV2::Timeout);
        }
        let timeout = Timespec::try_from(remaining)
            .map_err(|_| WorkerV3VerificationClientErrorV2::DeadlineOverflow)?;
        let mut descriptors = [PollFd::new(peer, wanted | PollFlags::ERR | PollFlags::HUP)];
        match poll(&mut descriptors, Some(&timeout)) {
            Ok(0) => return Err(WorkerV3VerificationClientErrorV2::Timeout),
            Ok(_) => {
                let ready = descriptors[0].revents();
                if ready.contains(PollFlags::NVAL) {
                    return Err(WorkerV3VerificationClientErrorV2::InvalidPeer);
                }
                if ready.contains(wanted) {
                    return Ok(());
                }
                if ready.contains(PollFlags::ERR) {
                    return Err(WorkerV3VerificationClientErrorV2::PeerFailed);
                }
                if ready.contains(PollFlags::HUP) {
                    return Err(WorkerV3VerificationClientErrorV2::PeerClosed);
                }
            }
            Err(rustix::io::Errno::INTR) => {}
            Err(source) => return Err(WorkerV3VerificationClientErrorV2::Poll(source.into())),
        }
    }
}

fn require_deadline(deadline: Instant) -> Result<(), WorkerV3VerificationClientErrorV2> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(WorkerV3VerificationClientErrorV2::Timeout);
    }
    Timespec::try_from(remaining)
        .map(|_| ())
        .map_err(|_| WorkerV3VerificationClientErrorV2::DeadlineOverflow)
}

fn descriptor_error(
    operation: &'static str,
    source: io::Error,
) -> WorkerV3VerificationClientErrorV2 {
    WorkerV3VerificationClientErrorV2::Descriptor { operation, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsFd, AsRawFd};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    use rustix::net::{
        SocketFlags, accept_with, bind, connect, listen, send, socket_with, socketpair,
    };

    fn pair() -> (OwnedFd, OwnedFd) {
        socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
            None,
        )
        .unwrap()
    }

    fn accepted_pair() -> (tempfile::TempDir, PathBuf, OwnedFd, OwnedFd) {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("service.sock");
        let address = SocketAddrUnix::new(&path).unwrap();
        let listener = socket_with(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC | SocketFlags::NONBLOCK,
            None,
        )
        .unwrap();
        bind(&listener, &address).unwrap();
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
        (root, path, client, service)
    }

    #[test]
    fn connected_path_admission_enforces_address_roles_and_response_credentials() {
        let (_root, path, client, service) = accepted_pair();
        let admitted = WorkerV3VerificationClientV2::admit_connected_path(
            client,
            &path,
            Duration::from_secs(1),
        )
        .unwrap();
        let expected = admitted.expected_response_credentials.unwrap();
        assert_eq!(expected.pid, std::process::id());
        assert_eq!(send(&service, b"x", SendFlags::NOSIGNAL).unwrap(), 1);
        assert_eq!(
            receive_packet(
                &admitted.peer,
                1,
                1,
                Instant::now() + Duration::from_secs(1),
                Some(expected),
            )
            .unwrap(),
            b"x"
        );
    }

    #[test]
    fn connected_path_admission_rejects_socketpair_and_retains_endpoint() {
        let (client, _service) = pair();
        let original = client.as_raw_fd();
        let failure = WorkerV3VerificationClientV2::admit_connected_path(
            client,
            Path::new("/run/fe2o3/service.sock"),
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(matches!(
            failure.source_error(),
            WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer
        ));
        let retained = failure.into_peer();
        assert_eq!(retained.as_raw_fd(), original);
        rustix::io::fcntl_getfd(&retained).unwrap();
    }

    #[test]
    fn connected_path_validator_rejects_empty_relative_noncanonical_and_overlong_names() {
        assert!(canonical_filesystem_unix_address(Path::new("/run/fe2o3/service.sock")).is_some());

        for path in [
            "",
            "relative.sock",
            "/run//service.sock",
            "/run/./service.sock",
            "/run/../service.sock",
            "/run/service.sock/",
        ] {
            assert!(
                canonical_filesystem_unix_address(Path::new(path)).is_none(),
                "{path}"
            );
        }

        let overlong = format!("/{}", "x".repeat(107));
        assert!(canonical_filesystem_unix_address(Path::new(&overlong)).is_none());

        let nul = Path::new(std::ffi::OsStr::from_bytes(b"/run/fe2o3/\0service.sock"));
        assert!(canonical_filesystem_unix_address(nul).is_none());
    }

    #[test]
    fn connected_path_admission_rejects_another_canonical_service_path() {
        let (_root, path, client, _service) = accepted_pair();
        let wrong_path = path.with_file_name("other.sock");
        let failure = WorkerV3VerificationClientV2::admit_connected_path(
            client,
            &wrong_path,
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(matches!(
            failure.source_error(),
            WorkerV3VerificationClientErrorV2::InvalidConnectedPathPeer
        ));
    }

    #[test]
    fn connected_path_response_from_a_handed_off_endpoint_is_rejected() {
        let (_root, path, client, service) = accepted_pair();
        let admitted = WorkerV3VerificationClientV2::admit_connected_path(
            client,
            &path,
            Duration::from_secs(5),
        )
        .unwrap();
        let expected = admitted.expected_response_credentials.unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("client_v2::tests::cross_process_connected_path_response_sender_helper")
            .arg("--ignored")
            .env("FE2O3_WORKER_V3_CONNECTED_PATH_RESPONSE_CHILD", "1")
            .stdin(Stdio::from(service))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        assert_ne!(child.id(), expected.pid);
        let result = receive_packet(
            &admitted.peer,
            1,
            1,
            Instant::now() + Duration::from_secs(2),
            Some(expected),
        );
        assert!(
            matches!(
                result,
                Err(WorkerV3VerificationClientErrorV2::ResponseCredentialsMismatch)
            ),
            "unexpected handed-off response result: {result:?}"
        );
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "response child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    #[ignore = "subprocess helper for the connected-path response credential test"]
    fn cross_process_connected_path_response_sender_helper() {
        if std::env::var_os("FE2O3_WORKER_V3_CONNECTED_PATH_RESPONSE_CHILD").is_none() {
            return;
        }
        let stdin = std::io::stdin();
        assert_eq!(send(&stdin, b"x", SendFlags::NOSIGNAL).unwrap(), 1);
    }

    #[test]
    fn non_begin_receive_rejects_ancillary_and_oversize_packets() {
        let (receiver, sender) = pair();
        let pipe = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC).unwrap();
        let descriptors = [pipe.0.as_fd()];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut ancillary = SendAncillaryBuffer::new(&mut space);
        assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
        assert_eq!(
            rustix::net::sendmsg(
                &sender,
                &[IoSlice::new(b"x")],
                &mut ancillary,
                SendFlags::NOSIGNAL,
            )
            .unwrap(),
            1
        );
        assert!(matches!(
            receive_packet(
                &receiver,
                1,
                1,
                Instant::now() + Duration::from_secs(1),
                None,
            ),
            Err(WorkerV3VerificationClientErrorV2::UnexpectedAncillaryData)
        ));

        let (receiver, sender) = pair();
        assert_eq!(
            rustix::net::send(&sender, b"xx", SendFlags::NOSIGNAL).unwrap(),
            2
        );
        assert!(matches!(
            receive_packet(
                &receiver,
                1,
                1,
                Instant::now() + Duration::from_secs(1),
                None,
            ),
            Err(WorkerV3VerificationClientErrorV2::PacketOversize {
                maximum: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn terminal_boundary_requires_exact_peer_eof() {
        let (receiver, sender) = pair();
        assert_eq!(
            rustix::net::send(&sender, b"trailing", SendFlags::NOSIGNAL).unwrap(),
            8
        );
        assert!(matches!(
            require_peer_eof(&receiver, Instant::now() + Duration::from_secs(1)),
            Err(WorkerV3VerificationClientErrorV2::TrailingTransfer)
        ));

        let (receiver, _sender) = pair();
        assert!(matches!(
            require_peer_eof(&receiver, Instant::now() + Duration::from_millis(10)),
            Err(WorkerV3VerificationClientErrorV2::Timeout)
        ));
    }
}
