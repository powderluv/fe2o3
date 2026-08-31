//! Worker V3 authority join for the sole direct-KFD production transition.

use core::{fmt, time::Duration};

use fe2o3_kfd::{
    CheckedGfx942XnackMinusDevice, HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
    KfdCooperativeTargetTelemetryEndpointV1, KfdTargetDebugAllocationPhaseV1,
    KfdTargetDebugArtifactIdentityV1, KfdTargetDebugArtifactRoleV1, KfdTargetDebugDispatchPhaseV1,
    KfdTargetDebugMemoryAccessV1, KfdTargetDebugMemoryKindV1, KfdTargetDebugSessionOutcomeV1,
    KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryPayloadV1,
    KfdTargetDebugTelemetryTransportErrorV1, KfdTargetRuntimeDebugTokenV1,
    execute_gfx942_kfd_debug_target_dispatch_unchecked_v1,
    execute_gfx942_kfd_dispatch_unchecked_v1,
};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::Gfx942RuntimeDispatchBufferKindV1;
use crate::{
    Gfx942RuntimeBufferAccessV1, Gfx942RuntimePreparedBufferPolicyV1,
    PreparedGfx942RuntimeDispatchV1,
};

/// Authenticated, invocation-specific Worker V3 authority for one gfx942 dispatch.
///
/// The trait transports authority established elsewhere; it does not define another verifier or
/// permit the runtime to infer authority from descriptive hashes. The safe runtime transition
/// independently compares every returned identity to its private prepared request and checked KFD
/// device before entering native mechanics.
///
/// Safe application code cannot implement this boundary:
///
/// ```compile_fail
/// use fe2o3_runtime::WorkerV3Gfx942ExecutionAuthorityV1;
///
/// struct ForgedAuthority;
///
/// impl WorkerV3Gfx942ExecutionAuthorityV1 for ForgedAuthority {
///     type CurrentnessError = core::convert::Infallible;
///
///     fn finalized_hsaco_sha256(&self) -> [u8; 32] { [0; 32] }
///     fn finalized_hsaco_length(&self) -> u64 { 0 }
///     fn kernel_name(&self) -> &str { "forged" }
///     fn dispatch_contract_sha256(&self) -> [u8; 32] { [0; 32] }
///     fn device_unique_id(&self) -> u64 { 0 }
///     fn revalidate_currentness(&self) -> Result<(), Self::CurrentnessError> { Ok(()) }
/// }
/// ```
///
/// # Safety
///
/// Implementations must be emitted only after one reviewed Worker V3 verifier has authenticated
/// the exact compiler lineage, finalized artifact, generated Rust ABI and effect contract,
/// machine effects, and proof-to-executable binding. The verifier may establish a universally
/// quantified kernel theorem, but a trusted composition boundary must then instantiate it with
/// the exact invocation arguments, launch geometry, alias and race discipline, bounds,
/// initialization, and completion policy represented by the returned dispatch-contract identity.
/// `device_unique_id` must identify the same checked KFD device retained by that composition.
/// `revalidate_currentness` must retain and recheck the same publication and evidence custody
/// through the call. A false implementation can make safe code execute unauthorised native GPU
/// memory accesses.
pub unsafe trait WorkerV3Gfx942ExecutionAuthorityV1 {
    type CurrentnessError;

    fn finalized_hsaco_sha256(&self) -> [u8; 32];

    fn finalized_hsaco_length(&self) -> u64;

    fn kernel_name(&self) -> &str;

    fn dispatch_contract_sha256(&self) -> [u8; 32];

    fn device_unique_id(&self) -> u64;

    fn revalidate_currentness(&self) -> Result<(), Self::CurrentnessError>;
}

/// Fail-closed rejection before native mutation or after confirmed completion and teardown.
#[derive(Debug)]
#[non_exhaustive]
pub enum Gfx942AuthorizedRuntimeExecutionErrorV1<E> {
    CurrentnessBeforeDispatch(E),
    CurrentnessAfterCompletion(E),
    ArtifactIdentityMismatch,
    ArtifactLengthMismatch,
    KernelNameMismatch,
    DispatchContractMismatch,
    DeviceIdentityMismatch,
    CompletedBufferCardinalityMismatch,
    CompletedBufferLengthMismatch { index: usize },
    ReadOnlyBufferModified { index: usize },
    Telemetry(KfdTargetDebugTelemetryTransportErrorV1),
}

impl<E: fmt::Display> fmt::Display for Gfx942AuthorizedRuntimeExecutionErrorV1<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentnessBeforeDispatch(error) => {
                write!(
                    formatter,
                    "Worker V3 currentness failed before dispatch: {error}"
                )
            }
            Self::CurrentnessAfterCompletion(error) => {
                write!(
                    formatter,
                    "Worker V3 currentness failed after completion: {error}"
                )
            }
            Self::ArtifactIdentityMismatch => {
                formatter.write_str("Worker V3 finalized-artifact identity mismatch")
            }
            Self::ArtifactLengthMismatch => {
                formatter.write_str("Worker V3 finalized-artifact length mismatch")
            }
            Self::KernelNameMismatch => {
                formatter.write_str("Worker V3 selected-kernel name mismatch")
            }
            Self::DispatchContractMismatch => {
                formatter.write_str("Worker V3 invocation contract mismatch")
            }
            Self::DeviceIdentityMismatch => {
                formatter.write_str("Worker V3 KFD device identity mismatch")
            }
            Self::CompletedBufferCardinalityMismatch => {
                formatter.write_str("KFD completion returned the wrong buffer cardinality")
            }
            Self::CompletedBufferLengthMismatch { index } => {
                write!(formatter, "KFD completion changed buffer {index} length")
            }
            Self::ReadOnlyBufferModified { index } => {
                write!(
                    formatter,
                    "KFD completion modified read-only buffer {index}"
                )
            }
            Self::Telemetry(error) => write!(formatter, "cooperative debug telemetry: {error}"),
        }
    }
}

impl<E> std::error::Error for Gfx942AuthorizedRuntimeExecutionErrorV1<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CurrentnessBeforeDispatch(error) | Self::CurrentnessAfterCompletion(error) => {
                Some(error)
            }
            Self::Telemetry(error) => Some(error),
            _ => None,
        }
    }
}

/// Target-owned endpoint and admitted process identities for one cooperative debug session.
///
/// The endpoint sends declarations only. The executable and process-instance identities are
/// supplied by the debugger's exact-bound launcher and do not authorize execution. Kernel,
/// dispatch, geometry, and allocation records are derived inside the runtime from the prepared
/// request that independently matches Worker V3 authority.
#[must_use = "a telemetry session must be consumed by a debug-target execution"]
pub struct AuthorizedRuntimeDebugTelemetrySessionV1 {
    endpoint: KfdCooperativeTargetTelemetryEndpointV1,
    process_instance: KfdTargetDebugTelemetryDigestV1,
    executable: KfdTargetDebugArtifactIdentityV1,
}

impl fmt::Debug for AuthorizedRuntimeDebugTelemetrySessionV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizedRuntimeDebugTelemetrySessionV1")
            .field("endpoint", &"telemetry-only")
            .field("process_instance", &self.process_instance)
            .field("executable", &self.executable)
            .finish_non_exhaustive()
    }
}

impl AuthorizedRuntimeDebugTelemetrySessionV1 {
    pub const fn new(
        endpoint: KfdCooperativeTargetTelemetryEndpointV1,
        process_instance: KfdTargetDebugTelemetryDigestV1,
        executable: KfdTargetDebugArtifactIdentityV1,
    ) -> Self {
        Self {
            endpoint,
            process_instance,
            executable,
        }
    }
}

/// One completed runtime buffer retaining its Worker V3 access classification.
#[derive(Debug, Eq, PartialEq)]
pub struct Gfx942AuthorizedRuntimeCompletedBufferV1 {
    bytes: Vec<u8>,
    access: Gfx942RuntimeBufferAccessV1,
    non_null_empty_slice_sentinel: bool,
}

impl Gfx942AuthorizedRuntimeCompletedBufferV1 {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub const fn access(&self) -> Gfx942RuntimeBufferAccessV1 {
        self.access
    }

    /// Reports a retained transport allocation for a logical zero-length generated slice.
    #[doc(hidden)]
    pub const fn is_non_null_empty_slice_sentinel_v1(&self) -> bool {
        self.non_null_empty_slice_sentinel
    }
}

/// Redacted successful result after confirmed completion, effect checks, and teardown.
#[must_use]
pub struct Gfx942AuthorizedRuntimeDispatchResultV1 {
    buffers: Vec<Gfx942AuthorizedRuntimeCompletedBufferV1>,
    packet_id: u64,
    queue_id: u32,
    completion_elapsed: Duration,
}

impl fmt::Debug for Gfx942AuthorizedRuntimeDispatchResultV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gfx942AuthorizedRuntimeDispatchResultV1")
            .field("buffers", &self.buffers.len())
            .field("packet_id", &self.packet_id)
            .field("queue_id", &self.queue_id)
            .field("completion_elapsed", &self.completion_elapsed)
            .finish()
    }
}

impl Gfx942AuthorizedRuntimeDispatchResultV1 {
    pub fn buffers(&self) -> &[Gfx942AuthorizedRuntimeCompletedBufferV1] {
        &self.buffers
    }

    pub fn into_buffers(self) -> Vec<Gfx942AuthorizedRuntimeCompletedBufferV1> {
        self.buffers
    }

    pub const fn packet_id(&self) -> u64 {
        self.packet_id
    }

    pub const fn queue_id(&self) -> u32 {
        self.queue_id
    }

    pub const fn completion_elapsed(&self) -> Duration {
        self.completion_elapsed
    }
}

/// Consumes one exact Worker V3 authority and executes its matching prepared KFD request.
///
/// Every authority and device mismatch fails before native mutation. Once KFD mutation starts, the
/// low-level transaction requires process termination on every error; this safe boundary enforces
/// that contract by aborting instead of returning an error into potentially live application code.
pub fn execute_authorized_gfx942_runtime_dispatch_v1<A>(
    authority: A,
    device: CheckedGfx942XnackMinusDevice,
    prepared: PreparedGfx942RuntimeDispatchV1,
) -> Result<
    Gfx942AuthorizedRuntimeDispatchResultV1,
    Gfx942AuthorizedRuntimeExecutionErrorV1<A::CurrentnessError>,
>
where
    A: WorkerV3Gfx942ExecutionAuthorityV1,
{
    authority
        .revalidate_currentness()
        .map_err(Gfx942AuthorizedRuntimeExecutionErrorV1::CurrentnessBeforeDispatch)?;
    validate_authority_v1(&authority, &device, &prepared)?;
    let (request, buffer_policies) = prepared.into_authorized_execution_parts();
    // SAFETY: the unsafe authority implementation promises the complete semantic obligations for
    // the exact identities independently compared above. Every native failure aborts immediately,
    // satisfying the mechanics transaction's terminal-failure contract.
    let result = match unsafe { execute_gfx942_kfd_dispatch_unchecked_v1(device, request) } {
        Ok(result) => result,
        Err(_) => std::process::abort(),
    };
    let result = validate_completed_buffers_v1(result, buffer_policies);
    authority
        .revalidate_currentness()
        .map_err(Gfx942AuthorizedRuntimeExecutionErrorV1::CurrentnessAfterCompletion)?;
    result.map_err(map_completed_buffer_error_v1)
}

/// Executes an exact Worker V3 request as a cooperative direct-KFD debug target.
///
/// Telemetry remains descriptive: it cannot create or replace Worker V3 authority. Every
/// telemetry record after the launcher-owned session start is derived from the same prepared
/// request compared by the runtime authority gate. A telemetry failure before the native
/// dispatch returns without entering queue mechanics. Once native execution starts, a KFD or
/// telemetry failure aborts because returning could expose an ambiguous device or debugger state.
pub fn execute_authorized_gfx942_runtime_debug_target_dispatch_v1<A>(
    authority: A,
    token: KfdTargetRuntimeDebugTokenV1,
    device: CheckedGfx942XnackMinusDevice,
    prepared: PreparedGfx942RuntimeDispatchV1,
    telemetry: Option<AuthorizedRuntimeDebugTelemetrySessionV1>,
) -> Result<
    Gfx942AuthorizedRuntimeDispatchResultV1,
    Gfx942AuthorizedRuntimeExecutionErrorV1<A::CurrentnessError>,
>
where
    A: WorkerV3Gfx942ExecutionAuthorityV1,
{
    authority
        .revalidate_currentness()
        .map_err(Gfx942AuthorizedRuntimeExecutionErrorV1::CurrentnessBeforeDispatch)?;
    validate_authority_v1(&authority, &device, &prepared)?;

    let telemetry_facts = telemetry
        .as_ref()
        .map(|_| DebugTelemetryFactsV1::from_prepared(&prepared))
        .transpose()
        .map_err(Gfx942AuthorizedRuntimeExecutionErrorV1::Telemetry)?;
    let mut telemetry = telemetry;
    if let (Some(session), Some(facts)) = (&mut telemetry, &telemetry_facts) {
        session
            .emit_before_dispatch(facts)
            .map_err(Gfx942AuthorizedRuntimeExecutionErrorV1::Telemetry)?;
    }

    let (request, buffer_policies) = prepared.into_authorized_execution_parts();
    // SAFETY: the exact authority is independently checked above. The debug token is the
    // reviewed current-process mode-1 KFD transition, and every native error remains terminal.
    let result = match unsafe {
        execute_gfx942_kfd_debug_target_dispatch_unchecked_v1(token, device, request)
    } {
        Ok(result) => result,
        Err(_) => std::process::abort(),
    };
    let result = validate_completed_buffers_v1(result, buffer_policies);

    if let (Some(session), Some(facts)) = (&mut telemetry, &telemetry_facts) {
        abort_on_telemetry_error(session.emit_completed_dispatch(facts));
    }
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            if let Some(session) = &mut telemetry {
                abort_on_telemetry_error(
                    session.emit_session_end(KfdTargetDebugSessionOutcomeV1::Failed),
                );
            }
            return Err(map_completed_buffer_error_v1(error));
        }
    };
    if let Err(error) = authority.revalidate_currentness() {
        if let Some(session) = &mut telemetry {
            abort_on_telemetry_error(
                session.emit_session_end(KfdTargetDebugSessionOutcomeV1::Failed),
            );
        }
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::CurrentnessAfterCompletion(error));
    }
    if let Some(session) = &mut telemetry {
        abort_on_telemetry_error(
            session.emit_session_end(KfdTargetDebugSessionOutcomeV1::Completed),
        );
    }
    Ok(result)
}

const DEBUG_KERNEL_IDENTITY_DOMAIN_V1: &[u8] = b"fe2o3.runtime.debug.kernel-identity.v1\0";
const DEBUG_LOGICAL_QUEUE_IDENTITY_DOMAIN_V1: &[u8] =
    b"fe2o3.runtime.debug.logical-queue-identity.v1\0";
const DEBUG_ALLOCATION_IDENTITY_DOMAIN_V1: &[u8] = b"fe2o3.runtime.debug.allocation-identity.v1\0";

#[derive(Clone, Copy)]
struct DebugTelemetryAllocationV1 {
    identity: KfdTargetDebugTelemetryDigestV1,
    byte_length: u64,
    access: KfdTargetDebugMemoryAccessV1,
    memory_kind: KfdTargetDebugMemoryKindV1,
    alignment: u64,
}

struct DebugTelemetryFactsV1 {
    code_object: KfdTargetDebugArtifactIdentityV1,
    dispatch: KfdTargetDebugTelemetryDigestV1,
    kernel: KfdTargetDebugTelemetryDigestV1,
    logical_queue: KfdTargetDebugTelemetryDigestV1,
    grid: [u32; 3],
    workgroup: [u32; 3],
    dynamic_shared_memory_bytes: u32,
    allocations: Vec<DebugTelemetryAllocationV1>,
}

impl DebugTelemetryFactsV1 {
    fn from_prepared(
        prepared: &PreparedGfx942RuntimeDispatchV1,
    ) -> Result<Self, KfdTargetDebugTelemetryTransportErrorV1> {
        let identity = prepared.identity();
        let code_object_digest = telemetry_digest(identity.object_sha256())?;
        let dispatch = telemetry_digest(prepared.dispatch_contract_sha256())?;
        let kernel = debug_kernel_identity_v1(identity.closure_sha256(), prepared.kernel_name())?;
        let logical_queue = debug_logical_queue_identity_v1(dispatch)?;
        let allocations = prepared
            .buffer_policies
            .iter()
            .enumerate()
            .map(|(ordinal, policy)| {
                let ordinal = u64::try_from(ordinal).expect("bounded buffer ordinal fits u64");
                let access = telemetry_access(policy.access());
                let memory_kind = KfdTargetDebugMemoryKindV1::HostVisible;
                let alignment = HOST_VISIBLE_MEMORY_PAGE_BYTES_V1;
                let identity = debug_allocation_identity_v1(
                    dispatch,
                    ordinal,
                    policy.byte_length(),
                    access,
                    memory_kind,
                    alignment,
                )?;
                Ok(DebugTelemetryAllocationV1 {
                    identity,
                    byte_length: policy.byte_length(),
                    access,
                    memory_kind,
                    alignment,
                })
            })
            .collect::<Result<Vec<_>, KfdTargetDebugTelemetryTransportErrorV1>>()?;
        let geometry = prepared.geometry();
        Ok(Self {
            code_object: KfdTargetDebugArtifactIdentityV1::new(
                code_object_digest,
                prepared.finalized_hsaco_length(),
            )?,
            dispatch,
            kernel,
            logical_queue,
            grid: geometry.grid(),
            workgroup: geometry.workgroup().map(u32::from),
            dynamic_shared_memory_bytes: prepared.dynamic_group_segment_bytes(),
            allocations,
        })
    }

    fn dispatch_payload(
        &self,
        phase: KfdTargetDebugDispatchPhaseV1,
    ) -> KfdTargetDebugTelemetryPayloadV1 {
        KfdTargetDebugTelemetryPayloadV1::Dispatch {
            phase,
            dispatch: self.dispatch,
            kernel: self.kernel,
            code_object: self.code_object.digest(),
            logical_queue: self.logical_queue,
            grid: self.grid,
            workgroup: self.workgroup,
            dynamic_shared_memory_bytes: self.dynamic_shared_memory_bytes,
        }
    }

    fn allocation_payloads(
        &self,
        phase: KfdTargetDebugAllocationPhaseV1,
    ) -> impl Iterator<Item = KfdTargetDebugTelemetryPayloadV1> + '_ {
        self.allocations.iter().map(move |allocation| {
            KfdTargetDebugTelemetryPayloadV1::Allocation {
                phase,
                memory_kind: allocation.memory_kind,
                access: allocation.access,
                allocation: allocation.identity,
                logical_scope: self.dispatch,
                byte_length: allocation.byte_length,
                alignment: allocation.alignment,
            }
        })
    }
}

impl AuthorizedRuntimeDebugTelemetrySessionV1 {
    fn emit_before_dispatch(
        &mut self,
        facts: &DebugTelemetryFactsV1,
    ) -> Result<(), KfdTargetDebugTelemetryTransportErrorV1> {
        self.endpoint
            .send(KfdTargetDebugTelemetryPayloadV1::SessionStarted {
                process_instance: self.process_instance,
                executable: self.executable,
            })?;
        self.endpoint
            .send(KfdTargetDebugTelemetryPayloadV1::Artifact {
                role: KfdTargetDebugArtifactRoleV1::CodeObject,
                ordinal: 0,
                artifact: facts.code_object,
            })?;
        self.endpoint
            .send(facts.dispatch_payload(KfdTargetDebugDispatchPhaseV1::Prepared))?;
        for payload in facts.allocation_payloads(KfdTargetDebugAllocationPhaseV1::Created) {
            self.endpoint.send(payload)?;
        }
        Ok(())
    }

    fn emit_completed_dispatch(
        &mut self,
        facts: &DebugTelemetryFactsV1,
    ) -> Result<(), KfdTargetDebugTelemetryTransportErrorV1> {
        self.endpoint
            .send(facts.dispatch_payload(KfdTargetDebugDispatchPhaseV1::Submitted))?;
        self.endpoint
            .send(facts.dispatch_payload(KfdTargetDebugDispatchPhaseV1::Completed))?;
        for payload in facts.allocation_payloads(KfdTargetDebugAllocationPhaseV1::Released) {
            self.endpoint.send(payload)?;
        }
        Ok(())
    }

    fn emit_session_end(
        &mut self,
        outcome: KfdTargetDebugSessionOutcomeV1,
    ) -> Result<(), KfdTargetDebugTelemetryTransportErrorV1> {
        self.endpoint
            .send(KfdTargetDebugTelemetryPayloadV1::SessionEnded { outcome })?;
        Ok(())
    }
}

fn telemetry_digest(
    bytes: [u8; 32],
) -> Result<KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryTransportErrorV1> {
    KfdTargetDebugTelemetryDigestV1::from_bytes(bytes).map_err(Into::into)
}

fn debug_kernel_identity_v1(
    closure_sha256: [u8; 32],
    kernel_name: &str,
) -> Result<KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryTransportErrorV1> {
    domain_digest(
        DEBUG_KERNEL_IDENTITY_DOMAIN_V1,
        &[closure_sha256.as_slice(), kernel_name.as_bytes()],
    )
}

fn debug_logical_queue_identity_v1(
    dispatch: KfdTargetDebugTelemetryDigestV1,
) -> Result<KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryTransportErrorV1> {
    domain_digest(
        DEBUG_LOGICAL_QUEUE_IDENTITY_DOMAIN_V1,
        &[dispatch.as_bytes()],
    )
}

fn debug_allocation_identity_v1(
    dispatch: KfdTargetDebugTelemetryDigestV1,
    ordinal: u64,
    byte_length: u64,
    access: KfdTargetDebugMemoryAccessV1,
    memory_kind: KfdTargetDebugMemoryKindV1,
    alignment: u64,
) -> Result<KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryTransportErrorV1> {
    domain_digest(
        DEBUG_ALLOCATION_IDENTITY_DOMAIN_V1,
        &[
            dispatch.as_bytes(),
            &ordinal.to_le_bytes(),
            &byte_length.to_le_bytes(),
            &(access as u16).to_le_bytes(),
            &(memory_kind as u16).to_le_bytes(),
            &alignment.to_le_bytes(),
        ],
    )
}

fn domain_digest(
    domain: &[u8],
    fields: &[&[u8]],
) -> Result<KfdTargetDebugTelemetryDigestV1, KfdTargetDebugTelemetryTransportErrorV1> {
    let mut digest = Sha256::new();
    digest.update(domain);
    for field in fields {
        digest.update(
            u64::try_from(field.len())
                .expect("bounded telemetry identity field length fits u64")
                .to_le_bytes(),
        );
        digest.update(field);
    }
    telemetry_digest(digest.finalize().into())
}

const fn telemetry_access(access: Gfx942RuntimeBufferAccessV1) -> KfdTargetDebugMemoryAccessV1 {
    match access {
        Gfx942RuntimeBufferAccessV1::ReadOnly => KfdTargetDebugMemoryAccessV1::ReadOnly,
        Gfx942RuntimeBufferAccessV1::WriteOnly => KfdTargetDebugMemoryAccessV1::WriteOnly,
        Gfx942RuntimeBufferAccessV1::ReadWrite => KfdTargetDebugMemoryAccessV1::ReadWrite,
    }
}

fn abort_on_telemetry_error(result: Result<(), KfdTargetDebugTelemetryTransportErrorV1>) {
    if result.is_err() {
        std::process::abort();
    }
}

enum CompletedBufferValidationErrorV1 {
    Cardinality,
    Length { index: usize },
    ReadOnlyModified { index: usize },
}

fn map_completed_buffer_error_v1<E>(
    error: CompletedBufferValidationErrorV1,
) -> Gfx942AuthorizedRuntimeExecutionErrorV1<E> {
    match error {
        CompletedBufferValidationErrorV1::Cardinality => {
            Gfx942AuthorizedRuntimeExecutionErrorV1::CompletedBufferCardinalityMismatch
        }
        CompletedBufferValidationErrorV1::Length { index } => {
            Gfx942AuthorizedRuntimeExecutionErrorV1::CompletedBufferLengthMismatch { index }
        }
        CompletedBufferValidationErrorV1::ReadOnlyModified { index } => {
            Gfx942AuthorizedRuntimeExecutionErrorV1::ReadOnlyBufferModified { index }
        }
    }
}

fn validate_completed_buffers_v1(
    result: fe2o3_kfd::Gfx942KfdDispatchResultV1,
    policies: Vec<Gfx942RuntimePreparedBufferPolicyV1>,
) -> Result<Gfx942AuthorizedRuntimeDispatchResultV1, CompletedBufferValidationErrorV1> {
    let packet_id = result.packet_id();
    let queue_id = result.queue_id();
    let completion_elapsed = result.completion_elapsed();
    let buffers = result.into_buffers();
    if buffers.len() != policies.len() {
        return Err(CompletedBufferValidationErrorV1::Cardinality);
    }
    let mut completed = Vec::with_capacity(buffers.len());
    for (index, (buffer, policy)) in buffers.into_iter().zip(policies).enumerate() {
        let bytes = buffer.into_bytes();
        if !completed_buffer_has_expected_length_v1(&policy, &bytes) {
            return Err(CompletedBufferValidationErrorV1::Length { index });
        }
        if !completed_buffer_satisfies_policy_v1(&policy, &bytes) {
            return Err(CompletedBufferValidationErrorV1::ReadOnlyModified { index });
        }
        completed.push(Gfx942AuthorizedRuntimeCompletedBufferV1 {
            bytes: completed_buffer_logical_bytes_v1(&policy, bytes),
            access: policy.access(),
            non_null_empty_slice_sentinel: policy.is_non_null_empty_slice_sentinel_v1(),
        });
    }
    Ok(Gfx942AuthorizedRuntimeDispatchResultV1 {
        buffers: completed,
        packet_id,
        queue_id,
        completion_elapsed,
    })
}

fn completed_buffer_logical_bytes_v1(
    policy: &Gfx942RuntimePreparedBufferPolicyV1,
    allocation_bytes: Vec<u8>,
) -> Vec<u8> {
    if policy.is_non_null_empty_slice_sentinel_v1() {
        Vec::new()
    } else {
        allocation_bytes
    }
}

fn completed_buffer_satisfies_policy_v1(
    policy: &Gfx942RuntimePreparedBufferPolicyV1,
    completed_bytes: &[u8],
) -> bool {
    policy
        .read_only_initial_bytes()
        .is_none_or(|initial| initial == completed_bytes)
}

fn completed_buffer_has_expected_length_v1(
    policy: &Gfx942RuntimePreparedBufferPolicyV1,
    completed_bytes: &[u8],
) -> bool {
    u64::try_from(completed_bytes.len()).ok() == Some(policy.byte_length())
}

fn validate_authority_v1<A>(
    authority: &A,
    device: &CheckedGfx942XnackMinusDevice,
    prepared: &PreparedGfx942RuntimeDispatchV1,
) -> Result<(), Gfx942AuthorizedRuntimeExecutionErrorV1<A::CurrentnessError>>
where
    A: WorkerV3Gfx942ExecutionAuthorityV1,
{
    validate_authority_bindings_v1(
        authority,
        prepared.identity().object_sha256(),
        prepared.finalized_hsaco_length(),
        prepared.kernel_name(),
        prepared.dispatch_contract_sha256(),
        device.observation().unique_id(),
    )
}

fn validate_authority_bindings_v1<A>(
    authority: &A,
    finalized_hsaco_sha256: [u8; 32],
    finalized_hsaco_length: u64,
    kernel_name: &str,
    dispatch_contract_sha256: [u8; 32],
    device_unique_id: u64,
) -> Result<(), Gfx942AuthorizedRuntimeExecutionErrorV1<A::CurrentnessError>>
where
    A: WorkerV3Gfx942ExecutionAuthorityV1,
{
    if authority.finalized_hsaco_sha256() != finalized_hsaco_sha256 {
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::ArtifactIdentityMismatch);
    }
    if authority.finalized_hsaco_length() != finalized_hsaco_length {
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::ArtifactLengthMismatch);
    }
    if authority.kernel_name() != kernel_name {
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::KernelNameMismatch);
    }
    if authority.dispatch_contract_sha256() != dispatch_contract_sha256 {
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::DispatchContractMismatch);
    }
    if authority.device_unique_id() != device_unique_id {
        return Err(Gfx942AuthorizedRuntimeExecutionErrorV1::DeviceIdentityMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_kfd::{
        KfdDebuggerTelemetryEndpointV1, KfdTargetDebugSessionNonceV1,
        KfdTargetDebugTelemetryProcessV1, create_kfd_target_debug_telemetry_channel_v1,
    };

    const TELEMETRY_ABORT_CASE: &str = "FE2O3_RUNTIME_TELEMETRY_ABORT_CASE";

    struct TestAuthorityV1 {
        object: [u8; 32],
        length: u64,
        kernel: &'static str,
        dispatch: [u8; 32],
        device: u64,
    }

    // SAFETY: this implementation is confined to pure identity-comparison unit tests and can
    // never reach a native device token or the execution function.
    unsafe impl WorkerV3Gfx942ExecutionAuthorityV1 for TestAuthorityV1 {
        type CurrentnessError = core::convert::Infallible;

        fn finalized_hsaco_sha256(&self) -> [u8; 32] {
            self.object
        }

        fn finalized_hsaco_length(&self) -> u64 {
            self.length
        }

        fn kernel_name(&self) -> &str {
            self.kernel
        }

        fn dispatch_contract_sha256(&self) -> [u8; 32] {
            self.dispatch
        }

        fn device_unique_id(&self) -> u64 {
            self.device
        }

        fn revalidate_currentness(&self) -> Result<(), Self::CurrentnessError> {
            Ok(())
        }
    }

    fn authority() -> TestAuthorityV1 {
        TestAuthorityV1 {
            object: [1; 32],
            length: 7_000,
            kernel: "kernel_v1",
            dispatch: [2; 32],
            device: 0x1234,
        }
    }

    fn validate(
        authority: &TestAuthorityV1,
    ) -> Result<(), Gfx942AuthorizedRuntimeExecutionErrorV1<core::convert::Infallible>> {
        validate_authority_bindings_v1(authority, [1; 32], 7_000, "kernel_v1", [2; 32], 0x1234)
    }

    #[test]
    fn exact_worker_v3_runtime_bindings_are_required() {
        assert!(validate(&authority()).is_ok());

        let mut changed = authority();
        changed.object[0] ^= 1;
        assert!(matches!(
            validate(&changed),
            Err(Gfx942AuthorizedRuntimeExecutionErrorV1::ArtifactIdentityMismatch)
        ));
        let mut changed = authority();
        changed.length += 1;
        assert!(matches!(
            validate(&changed),
            Err(Gfx942AuthorizedRuntimeExecutionErrorV1::ArtifactLengthMismatch)
        ));
        let mut changed = authority();
        changed.kernel = "other";
        assert!(matches!(
            validate(&changed),
            Err(Gfx942AuthorizedRuntimeExecutionErrorV1::KernelNameMismatch)
        ));
        let mut changed = authority();
        changed.dispatch[0] ^= 1;
        assert!(matches!(
            validate(&changed),
            Err(Gfx942AuthorizedRuntimeExecutionErrorV1::DispatchContractMismatch)
        ));
        let mut changed = authority();
        changed.device += 1;
        assert!(matches!(
            validate(&changed),
            Err(Gfx942AuthorizedRuntimeExecutionErrorV1::DeviceIdentityMismatch)
        ));
    }

    #[test]
    fn completed_read_only_buffers_must_preserve_every_byte() {
        let read_only = Gfx942RuntimePreparedBufferPolicyV1 {
            access: Gfx942RuntimeBufferAccessV1::ReadOnly,
            allocation_byte_length: 4,
            read_only_initial_bytes: Some(vec![1, 2, 3, 4]),
            kind: Gfx942RuntimeDispatchBufferKindV1::LogicalBytes,
        };
        assert!(completed_buffer_satisfies_policy_v1(
            &read_only,
            &[1, 2, 3, 4]
        ));
        assert!(!completed_buffer_satisfies_policy_v1(
            &read_only,
            &[1, 2, 0, 4]
        ));
        assert!(completed_buffer_has_expected_length_v1(
            &read_only,
            &[9, 8, 7, 6]
        ));
        assert!(!completed_buffer_has_expected_length_v1(
            &read_only,
            &[9, 8, 7]
        ));

        for access in [
            Gfx942RuntimeBufferAccessV1::WriteOnly,
            Gfx942RuntimeBufferAccessV1::ReadWrite,
        ] {
            let writable = Gfx942RuntimePreparedBufferPolicyV1 {
                access,
                allocation_byte_length: 3,
                read_only_initial_bytes: None,
                kind: Gfx942RuntimeDispatchBufferKindV1::LogicalBytes,
            };
            assert!(completed_buffer_satisfies_policy_v1(&writable, &[9, 8, 7]));
            assert!(!completed_buffer_has_expected_length_v1(&writable, &[9, 8]));
        }

        let error = map_completed_buffer_error_v1::<core::convert::Infallible>(
            CompletedBufferValidationErrorV1::Length { index: 2 },
        );
        assert!(matches!(
            error,
            Gfx942AuthorizedRuntimeExecutionErrorV1::CompletedBufferLengthMismatch { index: 2 }
        ));
    }

    #[test]
    fn empty_slice_sentinel_policy_has_no_logical_capacity_and_rejects_mutation() {
        let sentinel = Gfx942RuntimePreparedBufferPolicyV1 {
            access: Gfx942RuntimeBufferAccessV1::ReadOnly,
            allocation_byte_length: 1,
            read_only_initial_bytes: Some(vec![0]),
            kind: Gfx942RuntimeDispatchBufferKindV1::NonNullEmptySliceSentinel,
        };

        assert!(sentinel.is_non_null_empty_slice_sentinel_v1());
        assert!(completed_buffer_has_expected_length_v1(&sentinel, &[0]));
        assert!(completed_buffer_satisfies_policy_v1(&sentinel, &[0]));
        assert!(!completed_buffer_satisfies_policy_v1(&sentinel, &[1]));
        assert!(!completed_buffer_has_expected_length_v1(&sentinel, &[]));
        assert!(completed_buffer_logical_bytes_v1(&sentinel, vec![0]).is_empty());
    }

    fn telemetry_digest_for_test(seed: u8) -> KfdTargetDebugTelemetryDigestV1 {
        KfdTargetDebugTelemetryDigestV1::from_bytes([seed; 32]).unwrap()
    }

    fn telemetry_session_for_test() -> (
        KfdDebuggerTelemetryEndpointV1,
        AuthorizedRuntimeDebugTelemetrySessionV1,
    ) {
        let nonce = KfdTargetDebugSessionNonceV1::from_bytes([11; 32]).unwrap();
        let process = KfdTargetDebugTelemetryProcessV1::capture(std::process::id()).unwrap();
        let (debugger_fd, target_fd) = create_kfd_target_debug_telemetry_channel_v1().unwrap();
        let debugger = KfdDebuggerTelemetryEndpointV1::admit(debugger_fd, nonce, process).unwrap();
        let target =
            KfdCooperativeTargetTelemetryEndpointV1::admit(target_fd, nonce, process).unwrap();
        let executable =
            KfdTargetDebugArtifactIdentityV1::new(telemetry_digest_for_test(12), 8_192).unwrap();
        (
            debugger,
            AuthorizedRuntimeDebugTelemetrySessionV1::new(
                target,
                telemetry_digest_for_test(13),
                executable,
            ),
        )
    }

    fn telemetry_facts_for_test() -> DebugTelemetryFactsV1 {
        let code_object =
            KfdTargetDebugArtifactIdentityV1::new(telemetry_digest_for_test(14), 4_096).unwrap();
        DebugTelemetryFactsV1 {
            code_object,
            dispatch: telemetry_digest_for_test(15),
            kernel: telemetry_digest_for_test(16),
            logical_queue: telemetry_digest_for_test(17),
            grid: [256, 2, 1],
            workgroup: [64, 1, 1],
            dynamic_shared_memory_bytes: 512,
            allocations: vec![
                DebugTelemetryAllocationV1 {
                    identity: telemetry_digest_for_test(18),
                    byte_length: 1_024,
                    access: KfdTargetDebugMemoryAccessV1::ReadOnly,
                    memory_kind: KfdTargetDebugMemoryKindV1::HostVisible,
                    alignment: HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
                },
                DebugTelemetryAllocationV1 {
                    identity: telemetry_digest_for_test(19),
                    byte_length: 2_048,
                    access: KfdTargetDebugMemoryAccessV1::ReadWrite,
                    memory_kind: KfdTargetDebugMemoryKindV1::HostVisible,
                    alignment: HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
                },
            ],
        }
    }

    fn assert_dispatch_payload(
        payload: &KfdTargetDebugTelemetryPayloadV1,
        phase: KfdTargetDebugDispatchPhaseV1,
        facts: &DebugTelemetryFactsV1,
    ) {
        let KfdTargetDebugTelemetryPayloadV1::Dispatch {
            phase: observed_phase,
            dispatch,
            kernel,
            code_object,
            logical_queue,
            grid,
            workgroup,
            dynamic_shared_memory_bytes,
        } = payload
        else {
            panic!("expected a dispatch telemetry payload")
        };
        assert_eq!(*observed_phase, phase);
        assert_eq!(*dispatch, facts.dispatch);
        assert_eq!(*kernel, facts.kernel);
        assert_eq!(*code_object, facts.code_object.digest());
        assert_eq!(*logical_queue, facts.logical_queue);
        assert_eq!(*grid, facts.grid);
        assert_eq!(*workgroup, facts.workgroup);
        assert_eq!(
            *dynamic_shared_memory_bytes,
            facts.dynamic_shared_memory_bytes
        );
    }

    fn assert_allocation_payload(
        payload: &KfdTargetDebugTelemetryPayloadV1,
        phase: KfdTargetDebugAllocationPhaseV1,
        facts: &DebugTelemetryFactsV1,
        ordinal: usize,
    ) {
        let expected = facts.allocations[ordinal];
        let KfdTargetDebugTelemetryPayloadV1::Allocation {
            phase: observed_phase,
            memory_kind,
            access,
            allocation,
            logical_scope,
            byte_length,
            alignment,
        } = payload
        else {
            panic!("expected an allocation telemetry payload")
        };
        assert_eq!(*observed_phase, phase);
        assert_eq!(*memory_kind, expected.memory_kind);
        assert_eq!(*access, expected.access);
        assert_eq!(*allocation, expected.identity);
        assert_eq!(*logical_scope, facts.dispatch);
        assert_eq!(*byte_length, expected.byte_length);
        assert_eq!(*alignment, expected.alignment);
    }

    #[test]
    fn cooperative_debug_telemetry_emits_only_bounded_logical_records() {
        let (mut debugger, mut session) = telemetry_session_for_test();
        let facts = telemetry_facts_for_test();

        session.emit_before_dispatch(&facts).unwrap();
        let before = (0..5)
            .map(|_| debugger.receive().unwrap().payload().clone())
            .collect::<Vec<_>>();
        assert!(matches!(
            &before[0],
            KfdTargetDebugTelemetryPayloadV1::SessionStarted {
                process_instance,
                executable: observed,
            } if *process_instance == session.process_instance && *observed == session.executable
        ));
        assert!(matches!(
            &before[1],
            KfdTargetDebugTelemetryPayloadV1::Artifact {
                role: KfdTargetDebugArtifactRoleV1::CodeObject,
                ordinal: 0,
                artifact,
            } if *artifact == facts.code_object
        ));
        assert_dispatch_payload(&before[2], KfdTargetDebugDispatchPhaseV1::Prepared, &facts);
        assert_allocation_payload(
            &before[3],
            KfdTargetDebugAllocationPhaseV1::Created,
            &facts,
            0,
        );
        assert_allocation_payload(
            &before[4],
            KfdTargetDebugAllocationPhaseV1::Created,
            &facts,
            1,
        );
        session.emit_completed_dispatch(&facts).unwrap();
        session
            .emit_session_end(KfdTargetDebugSessionOutcomeV1::Completed)
            .unwrap();
        let after = (0..5)
            .map(|_| debugger.receive().unwrap().payload().clone())
            .collect::<Vec<_>>();
        assert_dispatch_payload(&after[0], KfdTargetDebugDispatchPhaseV1::Submitted, &facts);
        assert_dispatch_payload(&after[1], KfdTargetDebugDispatchPhaseV1::Completed, &facts);
        assert_allocation_payload(
            &after[2],
            KfdTargetDebugAllocationPhaseV1::Released,
            &facts,
            0,
        );
        assert_allocation_payload(
            &after[3],
            KfdTargetDebugAllocationPhaseV1::Released,
            &facts,
            1,
        );
        assert!(matches!(
            &after[4],
            KfdTargetDebugTelemetryPayloadV1::SessionEnded {
                outcome: KfdTargetDebugSessionOutcomeV1::Completed,
            }
        ));
        assert!(debugger.is_finished());
    }

    #[test]
    fn telemetry_identities_bind_every_generic_kernel_and_allocation_axis() {
        let dispatch = telemetry_digest_for_test(21);
        let identity = |dispatch, ordinal, length, access, memory_kind, alignment| {
            debug_allocation_identity_v1(dispatch, ordinal, length, access, memory_kind, alignment)
                .unwrap()
        };
        let baseline = identity(
            dispatch,
            0,
            64,
            KfdTargetDebugMemoryAccessV1::ReadOnly,
            KfdTargetDebugMemoryKindV1::HostVisible,
            HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
        );
        assert_ne!(
            baseline,
            identity(
                telemetry_digest_for_test(22),
                0,
                64,
                KfdTargetDebugMemoryAccessV1::ReadOnly,
                KfdTargetDebugMemoryKindV1::HostVisible,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
            )
        );
        assert_ne!(
            baseline,
            identity(
                dispatch,
                1,
                64,
                KfdTargetDebugMemoryAccessV1::ReadOnly,
                KfdTargetDebugMemoryKindV1::HostVisible,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
            )
        );
        assert_ne!(
            baseline,
            identity(
                dispatch,
                0,
                65,
                KfdTargetDebugMemoryAccessV1::ReadOnly,
                KfdTargetDebugMemoryKindV1::HostVisible,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
            )
        );
        assert_ne!(
            baseline,
            identity(
                dispatch,
                0,
                64,
                KfdTargetDebugMemoryAccessV1::ReadWrite,
                KfdTargetDebugMemoryKindV1::HostVisible,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
            )
        );
        assert_ne!(
            baseline,
            identity(
                dispatch,
                0,
                64,
                KfdTargetDebugMemoryAccessV1::ReadOnly,
                KfdTargetDebugMemoryKindV1::KernelArguments,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1,
            )
        );
        assert_ne!(
            baseline,
            identity(
                dispatch,
                0,
                64,
                KfdTargetDebugMemoryAccessV1::ReadOnly,
                KfdTargetDebugMemoryKindV1::HostVisible,
                HOST_VISIBLE_MEMORY_PAGE_BYTES_V1 * 2,
            )
        );
        assert_ne!(
            debug_kernel_identity_v1([23; 32], "kernel_a").unwrap(),
            debug_kernel_identity_v1([23; 32], "kernel_b").unwrap()
        );
        assert_ne!(
            debug_kernel_identity_v1([23; 32], "kernel_a").unwrap(),
            debug_kernel_identity_v1([24; 32], "kernel_a").unwrap()
        );
        assert_ne!(
            debug_logical_queue_identity_v1(dispatch).unwrap(),
            debug_logical_queue_identity_v1(telemetry_digest_for_test(22)).unwrap()
        );
        assert_ne!(
            baseline,
            domain_digest(b"different-domain\0", &[dispatch.as_bytes()]).unwrap()
        );
        assert!(telemetry_digest([0; 32]).is_err());
    }

    #[test]
    fn pre_native_telemetry_failure_is_returned_and_poisoned() {
        let (debugger, mut session) = telemetry_session_for_test();
        drop(debugger);
        assert!(
            session
                .emit_before_dispatch(&telemetry_facts_for_test())
                .is_err()
        );
        assert!(matches!(
            session.emit_session_end(KfdTargetDebugSessionOutcomeV1::Failed),
            Err(KfdTargetDebugTelemetryTransportErrorV1::Poisoned)
        ));
    }

    #[test]
    fn failed_session_end_is_explicit_and_terminal() {
        let (mut debugger, mut session) = telemetry_session_for_test();
        let facts = telemetry_facts_for_test();
        session.emit_before_dispatch(&facts).unwrap();
        session.emit_completed_dispatch(&facts).unwrap();
        session
            .emit_session_end(KfdTargetDebugSessionOutcomeV1::Failed)
            .unwrap();
        let mut last = None;
        while !debugger.is_finished() {
            last = Some(debugger.receive().unwrap().payload().clone());
        }
        assert!(matches!(
            last,
            Some(KfdTargetDebugTelemetryPayloadV1::SessionEnded {
                outcome: KfdTargetDebugSessionOutcomeV1::Failed,
            })
        ));
    }

    #[test]
    fn post_native_telemetry_failure_aborts() {
        if std::env::var_os(TELEMETRY_ABORT_CASE).is_some() {
            abort_on_telemetry_error(Err(
                KfdTargetDebugTelemetryTransportErrorV1::SessionFinished,
            ));
            std::process::exit(99);
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "authorized_execution::tests::post_native_telemetry_failure_aborts",
                "--nocapture",
            ])
            .env(TELEMETRY_ABORT_CASE, "1")
            .output()
            .unwrap();
        assert_ne!(
            output.status.code(),
            Some(99),
            "post-native telemetry failure returned instead of aborting"
        );
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                output.status.signal(),
                Some(6),
                "post-native telemetry failure did not terminate with SIGABRT"
            );
        }
    }
}
