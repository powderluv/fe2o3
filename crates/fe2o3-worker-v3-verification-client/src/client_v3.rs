//! Explicit pathname V3 readiness. Legacy V2 APIs do not enter these states.
use crate::{
    WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationClientAdmissionFailureV2,
    WorkerV3VerificationClientErrorV2, WorkerV3VerificationClientV2,
    WorkerV3VerificationPayloadSnapshotsV1,
};
use fe2o3_worker_v3_verification_protocol::{
    WorkerV3VerificationProtocolErrorV3, WorkerV3VerificationReadyFrameV3,
    WorkerV3VerificationRequestV1,
};
use std::{fmt, os::fd::OwnedFd, path::Path, time::Instant};

/// Owned client which can only wait for Ready; it has no send operation.
/// ```compile_fail
/// use fe2o3_worker_v3_verification_client::WorkerV3VerificationClientV3;
/// fn duplicate(x: WorkerV3VerificationClientV3) { let _ = x.clone(); }
/// ```
pub struct WorkerV3VerificationClientV3 {
    inner: WorkerV3VerificationClientV2,
}
impl WorkerV3VerificationClientV3 {
    /// Admits the strict unnamed client endpoint without sending a Hello or Begin.
    pub fn admit_connected_path_until(
        peer: OwnedFd,
        path: &Path,
        deadline: Instant,
    ) -> Result<Self, WorkerV3VerificationClientAdmissionFailureV2> {
        WorkerV3VerificationClientV2::admit_connected_path_until(peer, path, deadline)
            .map(|inner| Self { inner })
    }
    /// The unchanged absolute deadline covers readiness and all later V2 phases.
    pub fn deadline(&self) -> Instant {
        self.inner.deadline()
    }
    /// Readiness is transport coordination, never service verification authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
    /// Waits for one exact credential-checked ReadyV3 packet. Errors close the endpoint.
    pub fn wait_ready(
        self,
    ) -> Result<WorkerV3VerificationReadyClientV3, WorkerV3VerificationClientErrorV3> {
        let bytes = self
            .inner
            .receive_readiness()
            .map_err(WorkerV3VerificationClientErrorV3::V2)?;
        let ready = WorkerV3VerificationReadyFrameV3::decode_canonical(&bytes)
            .map_err(WorkerV3VerificationClientErrorV3::Protocol)?;
        self.inner
            .require_readiness_quiet()
            .map_err(WorkerV3VerificationClientErrorV3::V2)?;
        Ok(WorkerV3VerificationReadyClientV3 {
            inner: self.inner,
            ready,
        })
    }
}
impl fmt::Debug for WorkerV3VerificationClientV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerV3VerificationClientV3")
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}
/// Move-only permission to send one nonce-bound Begin, not an application authorization.
/// ```compile_fail
/// use fe2o3_worker_v3_verification_client::WorkerV3VerificationReadyClientV3;
/// fn duplicate(x: WorkerV3VerificationReadyClientV3) { let _ = x.clone(); }
/// ```
pub struct WorkerV3VerificationReadyClientV3 {
    inner: WorkerV3VerificationClientV2,
    ready: WorkerV3VerificationReadyFrameV3,
}
impl WorkerV3VerificationReadyClientV3 {
    /// Returns the original absolute deadline.
    pub fn deadline(&self) -> Instant {
        self.inner.deadline()
    }
    /// This capability cannot load, launch, verify, or access a signer.
    pub const fn grants_authority(&self) -> bool {
        false
    }
    /// Consumes the nonce and endpoint once, then uses unchanged V2 response/current phases.
    pub fn begin(
        self,
        request: WorkerV3VerificationRequestV1,
        snapshots: WorkerV3VerificationPayloadSnapshotsV1,
    ) -> Result<WorkerV3VerificationBeginOutcomeV2, WorkerV3VerificationClientErrorV3> {
        self.inner
            .require_readiness_quiet()
            .map_err(WorkerV3VerificationClientErrorV3::V2)?;
        self.inner
            .begin_with_readiness(request, snapshots, Some(self.ready))
            .map_err(WorkerV3VerificationClientErrorV3::V2)
    }
}
impl fmt::Debug for WorkerV3VerificationReadyClientV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerV3VerificationReadyClientV3")
            .field("authority", &"none")
            .finish_non_exhaustive()
    }
}
/// V3 readiness/Begin refusal. Owned endpoints and ancillary descriptors close on error.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationClientErrorV3 {
    /// Original bounded credential-checked transport error.
    V2(WorkerV3VerificationClientErrorV2),
    /// Wrong or noncanonical readiness framing.
    Protocol(WorkerV3VerificationProtocolErrorV3),
}
impl fmt::Display for WorkerV3VerificationClientErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V3 readiness client refusal: {self:?}")
    }
}
impl std::error::Error for WorkerV3VerificationClientErrorV3 {}
