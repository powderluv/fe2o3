//! Unsafe, address-sealed composition for one direct-KFD gfx942 dispatch.
//!
//! This module owns only native mechanics. A safe caller must first obtain an
//! admitting Worker V3 decision that authenticates the code object, ABI,
//! resource requirements, memory effects, launch geometry, and completion
//! protocol for the exact request.

use core::fmt;
use std::time::{Duration, Instant};

use fe2o3_aql::{
    AMD_SIGNAL_ALIGNMENT_V1, AMD_SIGNAL_BYTES_V1, AQL_MIN_RING_BYTES_V1,
    AqlCompletionObservationV1, AqlDispatchGeometryV1, AqlKernelDispatchPacketV1,
    AqlPreparedKernelDispatchV1, ObservedGpuAddressV1, classify_acquired_completion_value_v1,
    initialize_pending_completion_signal_bytes_v1,
};

use super::{
    ComputeAqlQueueDestroyedV1, ComputeAqlQueueSessionErrorV1, ComputeAqlQueueSessionV1,
    KfdTargetRuntimeDebugQueueV1, QueueExceptionWaitObservationV1,
};
use crate::queue_linux::LinuxKfdRuntimeEnabledV1;
use crate::shared_memory::{
    ExecutableGttV1, GttGpuAccessibleExecutableV1, GttGpuAccessibleMutableV1,
    HostVisibleCoherentGttV1, KernargGttV1, SharedGttAllocationV1, SharedGttMappedResourceFactsV1,
    SharedGttMemorySessionV1,
};
use crate::{CheckedGfx942XnackMinusDevice, KfdTargetRuntimeDebugTokenV1, KfdWithAdmittedUapi};

const KERNEL_DESCRIPTOR_BYTES: u64 = 64;
const POINTER_BYTES: usize = 8;
const MAX_BUFFERS_V1: usize = 57;
const MAX_POINTER_FIXUPS_V1: usize = 256;
const MAX_KERNARG_BYTES_V1: usize = 64 * 1024;
const MAX_EXECUTABLE_OR_BUFFER_BYTES_V1: usize = 1 << 31;
const MAX_GFX942_FLAT_WORKGROUP_SIZE_V1: u32 = 1024;
const MAX_GFX942_GROUP_SEGMENT_BYTES_V1: u32 = 64 * 1024;
const MAX_TIMEOUT_MILLISECONDS_V1: u32 = 60_000;

/// Canonical claim boundary for the first one-shot direct-KFD dispatch.
pub const GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_V1: &str = concat!(
    "profile=fe2o3-gfx942-one-shot-direct-kfd-dispatch-r3-v1\n",
    "target=gfx942:xnack-,KFD-1.18,linux-x86_64,little-endian\n",
    "input=owned-materialized-no-relocation-image,checked-image-relative-descriptor-offset,owned-kernarg-template,owned-host-visible-buffers,checked-pointer-fixups,checked-aql-geometry,zero-private-segment,bounded-total-static-plus-dynamic-group-segment,bounded-timeout\n",
    "addresses=private-same-vm-bindings-only,all-bo-cpu-vmas-identical-to-gpu-vas,no-fd-handle-mapping-pointer-or-gpu-address-export\n",
    "allocation=image:executable-gtt,kernarg:kernarg-gtt,signal-and-buffers:host-visible-coherent-gtt\n",
    "publication=one-private-single-producer-aql-packet,body-before-release-header,release-fenced-doorbell\n",
    "completion=exact-64-byte-user-signal,pending-1,acquire-poll,complete-0,unexpected-or-timeout-terminal;timeout-observes-aql-write-read-acquire-before-one-shot-zero-timeout-queue-exception-wait\n",
    "teardown=confirmed-queue-event-runtime-doorbell-and-queue-resource-destroy-before-output-readback,then-explicit-execution-unmap-and-release\n",
    "failure=post-vm-or-post-publication-error-requires-process-termination,no-drop-native-cleanup-or-retry\n",
    "authority=unsafe-mechanics-only,caller-must-supply-exact-Worker-V3-artifact-abi-effect-alias-bounds-geometry-and-quiescence-authority\n",
    "excluded=safe-launch-api,verifier-admission,arbitrary-hsaco-authority,persistent-queue,multi-packet,multi-process-recovery\n",
);

/// SHA-256 of [`GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_V1`].
pub const GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_SHA256_V1: &str =
    "d98a09d37b1d49ae878a146078d1fd05ed38fceb55bb2d866e6e7f3fba3a25ae";

/// One owned host-visible allocation supplied to a direct dispatch.
///
/// The complete byte vector is returned after confirmed completion and queue
/// teardown. Buffer meaning and access mode belong to the Worker V3 contract.
#[derive(Debug, Eq, PartialEq)]
pub struct Gfx942KfdDispatchBufferV1 {
    bytes: Vec<u8>,
}

impl Gfx942KfdDispatchBufferV1 {
    pub fn new(bytes: Vec<u8>) -> Result<Self, Gfx942KfdDispatchRequestErrorV1> {
        if bytes.is_empty() {
            return Err(Gfx942KfdDispatchRequestErrorV1::EmptyBuffer);
        }
        Ok(Self { bytes })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// One checked replacement of an eight-byte zero kernarg placeholder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx942KfdDispatchPointerFixupV1 {
    kernarg_offset: usize,
    buffer_index: usize,
    buffer_byte_offset: usize,
    required_alignment: u64,
}

impl Gfx942KfdDispatchPointerFixupV1 {
    pub const fn new(
        kernarg_offset: usize,
        buffer_index: usize,
        buffer_byte_offset: usize,
        required_alignment: u64,
    ) -> Self {
        Self {
            kernarg_offset,
            buffer_index,
            buffer_byte_offset,
            required_alignment,
        }
    }

    pub const fn kernarg_offset(self) -> usize {
        self.kernarg_offset
    }

    pub const fn buffer_index(self) -> usize {
        self.buffer_index
    }

    pub const fn buffer_byte_offset(self) -> usize {
        self.buffer_byte_offset
    }

    pub const fn required_alignment(self) -> u64 {
        self.required_alignment
    }
}

/// Structural rejection before any KFD operation or VM acquisition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Gfx942KfdDispatchRequestErrorV1 {
    EmptyExecutableImage,
    ExecutableImageTooLarge,
    DescriptorOffsetOverflow,
    DescriptorOutOfBounds,
    DescriptorMisaligned,
    EmptyKernarg,
    KernargTooLarge,
    InvalidKernargAlignment,
    EmptyBuffer,
    BufferTooLarge,
    TooManyBuffers,
    TooManyPointerFixups,
    PointerFixupOutOfBounds,
    PointerFixupMisaligned,
    PointerFixupBufferOutOfBounds,
    InvalidPointerAlignment,
    PointerTargetMisaligned,
    NonzeroPointerPlaceholder,
    DuplicatePointerFixup,
    WorkgroupTooLarge,
    PrivateSegmentUnsupported,
    GroupSegmentTooLarge,
    InvalidTimeout,
}

impl fmt::Display for Gfx942KfdDispatchRequestErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gfx942KfdDispatchRequestErrorV1 {}

/// Fully structurally checked but semantically unauthorised dispatch input.
///
/// Construction performs no native operation. The type deliberately cannot
/// authenticate executable semantics, ABI meaning, memory effects, aliasing,
/// or completion quiescence; those are the safety requirements of the unsafe
/// execution function.
#[must_use = "a checked request still requires Worker V3 authority before execution"]
pub struct Gfx942KfdDispatchRequestV1 {
    executable_image: Vec<u8>,
    descriptor_offset: u64,
    kernarg_template: Vec<u8>,
    kernarg_alignment: u64,
    buffers: Vec<Gfx942KfdDispatchBufferV1>,
    pointer_fixups: Vec<Gfx942KfdDispatchPointerFixupV1>,
    geometry: AqlDispatchGeometryV1,
    private_segment_size: u32,
    group_segment_size: u32,
    timeout_milliseconds: u32,
}

impl fmt::Debug for Gfx942KfdDispatchRequestV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gfx942KfdDispatchRequestV1")
            .field("executable_image_bytes", &self.executable_image.len())
            .field("descriptor_offset", &self.descriptor_offset)
            .field("kernarg_bytes", &self.kernarg_template.len())
            .field("kernarg_alignment", &self.kernarg_alignment)
            .field("buffers", &self.buffers.len())
            .field("pointer_fixups", &self.pointer_fixups.len())
            .field("geometry", &self.geometry)
            .field("private_segment_size", &self.private_segment_size)
            .field("group_segment_size", &self.group_segment_size)
            .field("timeout_milliseconds", &self.timeout_milliseconds)
            .finish()
    }
}

impl Gfx942KfdDispatchRequestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executable_image: Vec<u8>,
        descriptor_offset: u64,
        kernarg_template: Vec<u8>,
        kernarg_alignment: u64,
        buffers: Vec<Gfx942KfdDispatchBufferV1>,
        pointer_fixups: Vec<Gfx942KfdDispatchPointerFixupV1>,
        geometry: AqlDispatchGeometryV1,
        private_segment_size: u32,
        group_segment_size: u32,
        timeout_milliseconds: u32,
    ) -> Result<Self, Gfx942KfdDispatchRequestErrorV1> {
        validate_request(
            &executable_image,
            descriptor_offset,
            &kernarg_template,
            kernarg_alignment,
            &buffers,
            &pointer_fixups,
            geometry,
            private_segment_size,
            group_segment_size,
            timeout_milliseconds,
        )?;
        Ok(Self {
            executable_image,
            descriptor_offset,
            kernarg_template,
            kernarg_alignment,
            buffers,
            pointer_fixups,
            geometry,
            private_segment_size,
            group_segment_size,
            timeout_milliseconds,
        })
    }
}

/// Queue-exception evidence captured after a terminal completion timeout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx942KfdQueueExceptionObservationV1 {
    NoExceptionAtObservation,
    Exception { reason_bits: u64 },
    ObservationUnavailable,
}

impl Gfx942KfdQueueExceptionObservationV1 {
    fn from_private(observation: QueueExceptionWaitObservationV1) -> Self {
        match observation {
            QueueExceptionWaitObservationV1::NoExceptionAtObservation => {
                Self::NoExceptionAtObservation
            }
            QueueExceptionWaitObservationV1::Exception(reason) => Self::Exception {
                reason_bits: reason.get(),
            },
        }
    }
}

/// Terminal failure after a checked device has entered the one-shot native transaction.
#[derive(Debug)]
#[non_exhaustive]
pub enum Gfx942KfdDispatchErrorV1 {
    Preparation(ComputeAqlQueueSessionErrorV1),
    Submission,
    CompletionObservation(ComputeAqlQueueSessionErrorV1),
    UnexpectedCompletion(i64),
    CompletionTimeout {
        timeout_milliseconds: u32,
        packet_id: u64,
        queue_counters: Option<(u64, u64)>,
        queue_exception: Gfx942KfdQueueExceptionObservationV1,
    },
    Teardown(ComputeAqlQueueSessionErrorV1),
}

impl Gfx942KfdDispatchErrorV1 {
    /// Every execution error requires process termination. Native ambiguity,
    /// a live queue, or retained mapped memory may remain by design.
    pub const fn requires_process_termination(&self) -> bool {
        true
    }
}

impl fmt::Display for Gfx942KfdDispatchErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gfx942KfdDispatchErrorV1 {}

/// Redacted successful result after confirmed completion and explicit teardown.
#[must_use]
pub struct Gfx942KfdDispatchResultV1 {
    buffers: Vec<Gfx942KfdDispatchBufferV1>,
    packet_id: u64,
    queue_id: u32,
    completion_elapsed: Duration,
}

impl fmt::Debug for Gfx942KfdDispatchResultV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gfx942KfdDispatchResultV1")
            .field("buffers", &self.buffers.len())
            .field("packet_id", &self.packet_id)
            .field("queue_id", &self.queue_id)
            .field("completion_elapsed", &self.completion_elapsed)
            .finish()
    }
}

impl Gfx942KfdDispatchResultV1 {
    pub fn buffers(&self) -> &[Gfx942KfdDispatchBufferV1] {
        &self.buffers
    }

    pub fn into_buffers(self) -> Vec<Gfx942KfdDispatchBufferV1> {
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

struct PreparedDispatchResourcesV1 {
    executable: SharedGttAllocationV1<ExecutableGttV1, GttGpuAccessibleExecutableV1>,
    kernarg: SharedGttAllocationV1<KernargGttV1, GttGpuAccessibleMutableV1>,
    signal: SharedGttAllocationV1<HostVisibleCoherentGttV1, GttGpuAccessibleMutableV1>,
    buffers: Vec<SharedGttAllocationV1<HostVisibleCoherentGttV1, GttGpuAccessibleMutableV1>>,
    packet: Option<AqlPreparedKernelDispatchV1>,
}

/// Executes one structurally checked request through the private KFD/AQL owner.
///
/// # Safety
///
/// The caller must hold a fresh, exact Worker V3 admitting decision for this
/// request. In particular it must establish that the materialized image is the
/// authenticated no-relocation COV6 `gfx942:xnack-` image; the descriptor
/// offset, kernarg layout, pointer fixups, resource sizes, and geometry match
/// that image; every machine memory effect remains within the supplied
/// allocations with the admitted alias/race discipline; and completion value
/// zero is a system-scope quiescence point after which the kernel performs no
/// further access. The process must terminate after every returned error.
pub unsafe fn execute_gfx942_kfd_dispatch_unchecked_v1(
    device: CheckedGfx942XnackMinusDevice,
    request: Gfx942KfdDispatchRequestV1,
) -> Result<Gfx942KfdDispatchResultV1, Gfx942KfdDispatchErrorV1> {
    let timeout_milliseconds = request.timeout_milliseconds;
    let (session, resources) = device
        .create_compute_aql_queue_with(AQL_MIN_RING_BYTES_V1, move |memory| {
            prepare_dispatch_resources(memory, request)
        })
        .map_err(Gfx942KfdDispatchErrorV1::Preparation)?;

    execute_prepared_dispatch(
        session,
        resources,
        timeout_milliseconds,
        move |session, resources| {
            session.destroy_with(move |memory| release_dispatch_resources(memory, resources))
        },
    )
}

/// Executes the same one-shot dispatch transaction while carrying target-side
/// debug-runtime ownership through the existing queue and teardown path.
///
/// The caller-facing safety obligations are identical to
/// [`execute_gfx942_kfd_dispatch_unchecked_v1`]. The linear target token is
/// consumed so pre-handoff rejection can disable normally while event/queue
/// lifecycle mutation cannot duplicate or reanimate runtime authority.
///
/// # Safety
///
/// The caller must satisfy every exact code-image, ABI, memory-effect, alias,
/// geometry, and completion obligation documented by
/// [`execute_gfx942_kfd_dispatch_unchecked_v1`] for this request. The process
/// must terminate after every returned error.
pub unsafe fn execute_gfx942_kfd_debug_target_dispatch_unchecked_v1(
    mut token: KfdTargetRuntimeDebugTokenV1,
    device: CheckedGfx942XnackMinusDevice,
    request: Gfx942KfdDispatchRequestV1,
) -> Result<Gfx942KfdDispatchResultV1, Gfx942KfdDispatchErrorV1> {
    let (runtime, runtime_control) = token.queue_handoff_slots();
    execute_gfx942_kfd_debug_target_dispatch_with_runtime_unchecked_v1(
        device,
        request,
        runtime,
        runtime_control,
    )
}

fn execute_gfx942_kfd_debug_target_dispatch_with_runtime_unchecked_v1(
    device: CheckedGfx942XnackMinusDevice,
    request: Gfx942KfdDispatchRequestV1,
    runtime: &mut Option<LinuxKfdRuntimeEnabledV1>,
    runtime_control: &mut Option<KfdWithAdmittedUapi>,
) -> Result<Gfx942KfdDispatchResultV1, Gfx942KfdDispatchErrorV1> {
    let timeout_milliseconds = request.timeout_milliseconds;
    let (session, resources) = device
        .create_compute_aql_queue_for_debug_target_with(
            AQL_MIN_RING_BYTES_V1,
            move |memory| prepare_dispatch_resources(memory, request),
            runtime,
            runtime_control,
        )
        .map_err(Gfx942KfdDispatchErrorV1::Preparation)?;

    execute_prepared_dispatch(
        session,
        resources,
        timeout_milliseconds,
        move |session, resources| {
            let mut teardown = KfdTargetRuntimeDebugQueueV1::new(session).destroy()?;
            teardown.finish_with(move |memory| release_dispatch_resources(memory, resources))
        },
    )
}

fn execute_prepared_dispatch(
    mut session: ComputeAqlQueueSessionV1,
    mut resources: PreparedDispatchResourcesV1,
    timeout_milliseconds: u32,
    teardown: impl FnOnce(
        ComputeAqlQueueSessionV1,
        PreparedDispatchResourcesV1,
    ) -> Result<
        (ComputeAqlQueueDestroyedV1, Vec<Gfx942KfdDispatchBufferV1>),
        ComputeAqlQueueSessionErrorV1,
    >,
) -> Result<Gfx942KfdDispatchResultV1, Gfx942KfdDispatchErrorV1> {
    let packet = resources
        .packet
        .take()
        .expect("validated preparation retains one packet");
    let started = Instant::now();
    let packet_id = match session.submit_prepared(packet) {
        Ok(packet_id) => packet_id,
        Err(_) => {
            session.poison_terminal();
            return Err(Gfx942KfdDispatchErrorV1::Submission);
        }
    };
    let timeout = Duration::from_millis(u64::from(timeout_milliseconds));
    let mut polls = 0_u32;
    let completion_elapsed = loop {
        let value = match session.observe_dispatch_completion(&mut resources.signal) {
            Ok(value) => value,
            Err(error) => {
                session.poison_terminal();
                return Err(Gfx942KfdDispatchErrorV1::CompletionObservation(error));
            }
        };
        match classify_acquired_completion_value_v1(value) {
            AqlCompletionObservationV1::Completed => break started.elapsed(),
            AqlCompletionObservationV1::Unexpected(value) => {
                session.poison_terminal();
                return Err(Gfx942KfdDispatchErrorV1::UnexpectedCompletion(value));
            }
            AqlCompletionObservationV1::Pending => {}
        }
        if started.elapsed() >= timeout {
            let queue_counters = session.observe_dispatch_counters().ok();
            let queue_exception = session
                .observe_queue_exception(0)
                .map(Gfx942KfdQueueExceptionObservationV1::from_private)
                .unwrap_or(Gfx942KfdQueueExceptionObservationV1::ObservationUnavailable);
            return Err(Gfx942KfdDispatchErrorV1::CompletionTimeout {
                timeout_milliseconds,
                packet_id,
                queue_counters,
                queue_exception,
            });
        }
        polls = polls.wrapping_add(1);
        if polls.is_multiple_of(4096) {
            std::thread::yield_now();
        } else {
            core::hint::spin_loop();
        }
    };

    let (destroyed, buffers) =
        teardown(session, resources).map_err(Gfx942KfdDispatchErrorV1::Teardown)?;
    Ok(Gfx942KfdDispatchResultV1 {
        buffers,
        packet_id,
        queue_id: destroyed.queue_id(),
        completion_elapsed,
    })
}

impl ComputeAqlQueueSessionV1 {
    fn observe_dispatch_counters(&mut self) -> Result<(u64, u64), ComputeAqlQueueSessionErrorV1> {
        self.check_currentness()?;
        let engine = self
            .engine
            .as_mut()
            .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                "missing queue engine",
            ))?;
        let (backend, resources) = (&mut engine.backend, &mut engine.resources);
        let resource = resources
            .iter_mut()
            .find(|resource| resource.key == self.key)
            .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                "missing queue resources",
            ))?;
        let authority =
            resource
                .authority
                .as_mut()
                .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                    "released queue resources",
                ))?;
        backend
            .session
            .observe_aql_control_counters(&mut authority.control)
            .map_err(Into::into)
    }

    fn observe_dispatch_completion(
        &mut self,
        signal: &mut SharedGttAllocationV1<HostVisibleCoherentGttV1, GttGpuAccessibleMutableV1>,
    ) -> Result<i64, ComputeAqlQueueSessionErrorV1> {
        self.check_currentness()?;
        self.engine
            .as_mut()
            .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                "missing queue engine",
            ))?
            .backend
            .session
            .observe_completion_signal(signal)
            .map_err(Into::into)
    }
}

fn prepare_dispatch_resources(
    memory: &mut SharedGttMemorySessionV1,
    request: Gfx942KfdDispatchRequestV1,
) -> Result<PreparedDispatchResourcesV1, ComputeAqlQueueSessionErrorV1> {
    let mut buffers = Vec::with_capacity(request.buffers.len());
    for buffer in request.buffers {
        let bytes = buffer.into_bytes();
        let mut token = memory.allocate_host_visible_coherent(bytes.len())?;
        memory.with_bytes_mut(&mut token, |destination| {
            destination.copy_from_slice(&bytes)
        })?;
        buffers.push(token);
    }

    let mut kernarg = memory.allocate_kernarg(request.kernarg_template.len())?;
    memory.with_bytes_mut(&mut kernarg, |destination| {
        destination.copy_from_slice(&request.kernarg_template)
    })?;

    let mut signal = memory.allocate_host_visible_coherent(AMD_SIGNAL_BYTES_V1)?;
    memory.with_bytes_mut(&mut signal, |destination| {
        let destination: &mut [u8; AMD_SIGNAL_BYTES_V1] = destination
            .try_into()
            .expect("completion allocation has exact requested length");
        initialize_pending_completion_signal_bytes_v1(destination);
    })?;

    let mut executable = memory.allocate_executable(request.executable_image.len())?;
    memory.with_bytes_mut(&mut executable, |destination| {
        destination.copy_from_slice(&request.executable_image)
    })?;
    let executable = memory.seal_executable(executable)?;
    let executable = memory.map_executable_to_gpu(executable)?;

    let mut mapped_buffers = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        mapped_buffers.push(memory.map_to_gpu(buffer)?);
    }
    let buffer_facts = mapped_buffers
        .iter()
        .map(|buffer| memory.mapped_resource_facts(buffer))
        .collect::<Result<Vec<_>, _>>()?;
    memory.with_bytes_mut(&mut kernarg, |bytes| {
        patch_kernarg_pointers(bytes, &request.pointer_fixups, &buffer_facts)
    })??;
    let kernarg = memory.map_to_gpu(kernarg)?;
    let signal = memory.map_to_gpu(signal)?;

    let executable_facts = memory.mapped_resource_facts(&executable)?;
    let kernarg_facts = memory.mapped_resource_facts(&kernarg)?;
    let signal_facts = memory.mapped_resource_facts(&signal)?;
    let kernel_object = executable_facts
        .checked_gpu_subrange(
            request.descriptor_offset,
            KERNEL_DESCRIPTOR_BYTES,
            KERNEL_DESCRIPTOR_BYTES,
        )
        .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
            "executable descriptor binding",
        ))?;
    let kernarg_address = kernarg_facts
        .checked_gpu_subrange(
            0,
            u64::try_from(request.kernarg_template.len())
                .map_err(|_| ComputeAqlQueueSessionErrorV1::Contract("kernarg size conversion"))?,
            request.kernarg_alignment,
        )
        .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
            "kernarg address binding",
        ))?;
    let signal_address = signal_facts
        .checked_gpu_subrange(
            0,
            AMD_SIGNAL_BYTES_V1 as u64,
            AMD_SIGNAL_ALIGNMENT_V1 as u64,
        )
        .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
            "completion signal address binding",
        ))?;
    let packet = AqlKernelDispatchPacketV1::new_unpublished(
        request.geometry,
        request.private_segment_size,
        request.group_segment_size,
        ObservedGpuAddressV1::new(kernel_object).map_err(|_| {
            ComputeAqlQueueSessionErrorV1::Contract("kernel object address observation")
        })?,
        ObservedGpuAddressV1::new(kernarg_address)
            .map_err(|_| ComputeAqlQueueSessionErrorV1::Contract("kernarg address observation"))?,
        request.kernarg_alignment,
        ObservedGpuAddressV1::new(signal_address).map_err(|_| {
            ComputeAqlQueueSessionErrorV1::Contract("completion address observation")
        })?,
    )
    .map_err(|_| ComputeAqlQueueSessionErrorV1::Contract("AQL packet construction"))?;

    Ok(PreparedDispatchResourcesV1 {
        executable,
        kernarg,
        signal,
        buffers: mapped_buffers,
        packet: Some(packet),
    })
}

fn release_dispatch_resources(
    memory: &mut SharedGttMemorySessionV1,
    resources: PreparedDispatchResourcesV1,
) -> Result<Vec<Gfx942KfdDispatchBufferV1>, ComputeAqlQueueSessionErrorV1> {
    debug_assert!(resources.packet.is_none());
    let mut returned = Vec::with_capacity(resources.buffers.len());
    for buffer in resources.buffers {
        let buffer = memory.unmap_from_gpu(buffer)?;
        let bytes = memory.with_bytes(&buffer, |bytes| bytes.to_vec())?;
        memory.release(buffer)?;
        returned.push(Gfx942KfdDispatchBufferV1 { bytes });
    }
    let kernarg = memory.unmap_from_gpu(resources.kernarg)?;
    memory.release(kernarg)?;
    let signal = memory.unmap_from_gpu(resources.signal)?;
    memory.release(signal)?;
    let executable = memory.unmap_executable_from_gpu(resources.executable)?;
    memory.release_executable(executable)?;
    Ok(returned)
}

fn patch_kernarg_pointers(
    kernarg: &mut [u8],
    fixups: &[Gfx942KfdDispatchPointerFixupV1],
    buffers: &[SharedGttMappedResourceFactsV1],
) -> Result<(), ComputeAqlQueueSessionErrorV1> {
    for fixup in fixups {
        let facts =
            buffers
                .get(fixup.buffer_index)
                .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                    "pointer-fixup buffer binding",
                ))?;
        let address = facts
            .checked_gpu_subrange(
                u64::try_from(fixup.buffer_byte_offset).map_err(|_| {
                    ComputeAqlQueueSessionErrorV1::Contract("pointer-fixup offset conversion")
                })?,
                1,
                fixup.required_alignment,
            )
            .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
                "pointer-fixup target binding",
            ))?;
        patch_kernarg_pointer(kernarg, fixup, address)?;
    }
    Ok(())
}

fn patch_kernarg_pointer(
    kernarg: &mut [u8],
    fixup: &Gfx942KfdDispatchPointerFixupV1,
    address: u64,
) -> Result<(), ComputeAqlQueueSessionErrorV1> {
    if fixup.required_alignment == 0
        || !fixup.required_alignment.is_power_of_two()
        || address == 0
        || !address.is_multiple_of(fixup.required_alignment)
    {
        return Err(ComputeAqlQueueSessionErrorV1::Contract(
            "pointer-fixup non-null aligned address",
        ));
    }
    let end = fixup.kernarg_offset.checked_add(POINTER_BYTES).ok_or(
        ComputeAqlQueueSessionErrorV1::Contract("pointer-fixup kernarg range"),
    )?;
    kernarg
        .get_mut(fixup.kernarg_offset..end)
        .ok_or(ComputeAqlQueueSessionErrorV1::Contract(
            "pointer-fixup kernarg binding",
        ))?
        .copy_from_slice(&address.to_le_bytes());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_request(
    executable_image: &[u8],
    descriptor_offset: u64,
    kernarg_template: &[u8],
    kernarg_alignment: u64,
    buffers: &[Gfx942KfdDispatchBufferV1],
    pointer_fixups: &[Gfx942KfdDispatchPointerFixupV1],
    geometry: AqlDispatchGeometryV1,
    private_segment_size: u32,
    group_segment_size: u32,
    timeout_milliseconds: u32,
) -> Result<(), Gfx942KfdDispatchRequestErrorV1> {
    if executable_image.is_empty() {
        return Err(Gfx942KfdDispatchRequestErrorV1::EmptyExecutableImage);
    }
    if executable_image.len() > MAX_EXECUTABLE_OR_BUFFER_BYTES_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::ExecutableImageTooLarge);
    }
    if !descriptor_offset.is_multiple_of(KERNEL_DESCRIPTOR_BYTES) {
        return Err(Gfx942KfdDispatchRequestErrorV1::DescriptorMisaligned);
    }
    let descriptor_end = descriptor_offset
        .checked_add(KERNEL_DESCRIPTOR_BYTES)
        .ok_or(Gfx942KfdDispatchRequestErrorV1::DescriptorOffsetOverflow)?;
    if descriptor_end > executable_image.len() as u64 {
        return Err(Gfx942KfdDispatchRequestErrorV1::DescriptorOutOfBounds);
    }
    if kernarg_template.is_empty() {
        return Err(Gfx942KfdDispatchRequestErrorV1::EmptyKernarg);
    }
    if kernarg_template.len() > MAX_KERNARG_BYTES_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::KernargTooLarge);
    }
    if kernarg_alignment == 0 || kernarg_alignment > 4096 || !kernarg_alignment.is_power_of_two() {
        return Err(Gfx942KfdDispatchRequestErrorV1::InvalidKernargAlignment);
    }
    if buffers.len() > MAX_BUFFERS_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::TooManyBuffers);
    }
    for buffer in buffers {
        if buffer.bytes.is_empty() {
            return Err(Gfx942KfdDispatchRequestErrorV1::EmptyBuffer);
        }
        if buffer.bytes.len() > MAX_EXECUTABLE_OR_BUFFER_BYTES_V1 {
            return Err(Gfx942KfdDispatchRequestErrorV1::BufferTooLarge);
        }
    }
    if pointer_fixups.len() > MAX_POINTER_FIXUPS_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::TooManyPointerFixups);
    }
    let mut offsets = Vec::with_capacity(pointer_fixups.len());
    for fixup in pointer_fixups {
        let end = fixup
            .kernarg_offset
            .checked_add(POINTER_BYTES)
            .ok_or(Gfx942KfdDispatchRequestErrorV1::PointerFixupOutOfBounds)?;
        if end > kernarg_template.len() {
            return Err(Gfx942KfdDispatchRequestErrorV1::PointerFixupOutOfBounds);
        }
        if !fixup.kernarg_offset.is_multiple_of(POINTER_BYTES) {
            return Err(Gfx942KfdDispatchRequestErrorV1::PointerFixupMisaligned);
        }
        let buffer = buffers
            .get(fixup.buffer_index)
            .ok_or(Gfx942KfdDispatchRequestErrorV1::PointerFixupBufferOutOfBounds)?;
        if fixup.buffer_byte_offset >= buffer.bytes.len() {
            return Err(Gfx942KfdDispatchRequestErrorV1::PointerFixupBufferOutOfBounds);
        }
        if fixup.required_alignment == 0
            || fixup.required_alignment > 4096
            || !fixup.required_alignment.is_power_of_two()
        {
            return Err(Gfx942KfdDispatchRequestErrorV1::InvalidPointerAlignment);
        }
        if !(fixup.buffer_byte_offset as u64).is_multiple_of(fixup.required_alignment) {
            return Err(Gfx942KfdDispatchRequestErrorV1::PointerTargetMisaligned);
        }
        if kernarg_template[fixup.kernarg_offset..end]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(Gfx942KfdDispatchRequestErrorV1::NonzeroPointerPlaceholder);
        }
        offsets.push(fixup.kernarg_offset);
    }
    offsets.sort_unstable();
    if offsets.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Gfx942KfdDispatchRequestErrorV1::DuplicatePointerFixup);
    }
    let workgroup_size = geometry
        .workgroup()
        .into_iter()
        .map(u32::from)
        .product::<u32>();
    if workgroup_size > MAX_GFX942_FLAT_WORKGROUP_SIZE_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::WorkgroupTooLarge);
    }
    if private_segment_size != 0 {
        return Err(Gfx942KfdDispatchRequestErrorV1::PrivateSegmentUnsupported);
    }
    if group_segment_size > MAX_GFX942_GROUP_SEGMENT_BYTES_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::GroupSegmentTooLarge);
    }
    if timeout_milliseconds == 0 || timeout_milliseconds > MAX_TIMEOUT_MILLISECONDS_V1 {
        return Err(Gfx942KfdDispatchRequestErrorV1::InvalidTimeout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn valid_request() -> Result<Gfx942KfdDispatchRequestV1, Gfx942KfdDispatchRequestErrorV1> {
        Gfx942KfdDispatchRequestV1::new(
            vec![0; 128],
            64,
            vec![0; 16],
            16,
            vec![Gfx942KfdDispatchBufferV1::new(vec![0; 64])?],
            vec![Gfx942KfdDispatchPointerFixupV1::new(0, 0, 0, 4)],
            AqlDispatchGeometryV1::new([64, 1, 1], [64, 1, 1]).unwrap(),
            0,
            256,
            1_000,
        )
    }

    #[test]
    fn manifest_digest_is_current() {
        let actual = Sha256::digest(GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_V1)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(actual, GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_SHA256_V1);
    }

    #[test]
    fn terminal_exception_observation_preserves_admitted_reason_bits() {
        let reason = fe2o3_kfd_uapi::KfdQueueExceptionReasonV1::from_untrusted_wire(1 << 5)
            .expect("wave memory violation is admitted");
        assert_eq!(
            Gfx942KfdQueueExceptionObservationV1::from_private(
                QueueExceptionWaitObservationV1::Exception(reason),
            ),
            Gfx942KfdQueueExceptionObservationV1::Exception {
                reason_bits: 1 << 5,
            }
        );
        assert_eq!(
            Gfx942KfdQueueExceptionObservationV1::from_private(
                QueueExceptionWaitObservationV1::NoExceptionAtObservation,
            ),
            Gfx942KfdQueueExceptionObservationV1::NoExceptionAtObservation
        );
    }

    #[test]
    fn admits_one_complete_structural_request_without_native_work() {
        let request = valid_request().unwrap();
        assert_eq!(request.executable_image.len(), 128);
        assert_eq!(request.buffers.len(), 1);
    }

    #[test]
    fn rejects_empty_and_invalid_executable_ranges() {
        let mut request = valid_request().unwrap();
        request.executable_image.clear();
        assert_eq!(
            validate_request(
                &request.executable_image,
                request.descriptor_offset,
                &request.kernarg_template,
                request.kernarg_alignment,
                &request.buffers,
                &request.pointer_fixups,
                request.geometry,
                request.private_segment_size,
                request.group_segment_size,
                request.timeout_milliseconds,
            ),
            Err(Gfx942KfdDispatchRequestErrorV1::EmptyExecutableImage)
        );
        request.executable_image.resize(128, 0);
        assert_eq!(
            validate_request(
                &request.executable_image,
                65,
                &request.kernarg_template,
                request.kernarg_alignment,
                &request.buffers,
                &request.pointer_fixups,
                request.geometry,
                request.private_segment_size,
                request.group_segment_size,
                request.timeout_milliseconds,
            ),
            Err(Gfx942KfdDispatchRequestErrorV1::DescriptorMisaligned)
        );
        assert_eq!(
            validate_request(
                &request.executable_image,
                128,
                &request.kernarg_template,
                request.kernarg_alignment,
                &request.buffers,
                &request.pointer_fixups,
                request.geometry,
                request.private_segment_size,
                request.group_segment_size,
                request.timeout_milliseconds,
            ),
            Err(Gfx942KfdDispatchRequestErrorV1::DescriptorOutOfBounds)
        );
    }

    #[test]
    fn rejects_invalid_kernarg_and_timeout_bounds() {
        let request = valid_request().unwrap();
        for alignment in [0, 3, 8192] {
            assert_eq!(
                validate_request(
                    &request.executable_image,
                    request.descriptor_offset,
                    &request.kernarg_template,
                    alignment,
                    &request.buffers,
                    &request.pointer_fixups,
                    request.geometry,
                    request.private_segment_size,
                    request.group_segment_size,
                    request.timeout_milliseconds,
                ),
                Err(Gfx942KfdDispatchRequestErrorV1::InvalidKernargAlignment)
            );
        }
        for timeout in [0, MAX_TIMEOUT_MILLISECONDS_V1 + 1] {
            assert_eq!(
                validate_request(
                    &request.executable_image,
                    request.descriptor_offset,
                    &request.kernarg_template,
                    request.kernarg_alignment,
                    &request.buffers,
                    &request.pointer_fixups,
                    request.geometry,
                    request.private_segment_size,
                    request.group_segment_size,
                    timeout,
                ),
                Err(Gfx942KfdDispatchRequestErrorV1::InvalidTimeout)
            );
        }
    }

    #[test]
    fn rejects_unimplemented_scratch_and_out_of_profile_resources() {
        let request = valid_request().unwrap();
        let oversized_workgroup = AqlDispatchGeometryV1::new([1025, 1, 1], [1025, 1, 1]).unwrap();
        for (geometry, private, group, expected) in [
            (
                oversized_workgroup,
                0,
                0,
                Gfx942KfdDispatchRequestErrorV1::WorkgroupTooLarge,
            ),
            (
                request.geometry,
                1,
                0,
                Gfx942KfdDispatchRequestErrorV1::PrivateSegmentUnsupported,
            ),
            (
                request.geometry,
                0,
                MAX_GFX942_GROUP_SEGMENT_BYTES_V1 + 1,
                Gfx942KfdDispatchRequestErrorV1::GroupSegmentTooLarge,
            ),
        ] {
            assert_eq!(
                validate_request(
                    &request.executable_image,
                    request.descriptor_offset,
                    &request.kernarg_template,
                    request.kernarg_alignment,
                    &request.buffers,
                    &request.pointer_fixups,
                    geometry,
                    private,
                    group,
                    request.timeout_milliseconds,
                ),
                Err(expected)
            );
        }
    }

    #[test]
    fn rejects_every_pointer_fixup_substitution_class() {
        let request = valid_request().unwrap();
        let cases = [
            (
                Gfx942KfdDispatchPointerFixupV1::new(9, 0, 0, 4),
                Gfx942KfdDispatchRequestErrorV1::PointerFixupOutOfBounds,
            ),
            (
                Gfx942KfdDispatchPointerFixupV1::new(1, 0, 0, 4),
                Gfx942KfdDispatchRequestErrorV1::PointerFixupMisaligned,
            ),
            (
                Gfx942KfdDispatchPointerFixupV1::new(0, 1, 0, 4),
                Gfx942KfdDispatchRequestErrorV1::PointerFixupBufferOutOfBounds,
            ),
            (
                Gfx942KfdDispatchPointerFixupV1::new(0, 0, 64, 4),
                Gfx942KfdDispatchRequestErrorV1::PointerFixupBufferOutOfBounds,
            ),
            (
                Gfx942KfdDispatchPointerFixupV1::new(0, 0, 0, 3),
                Gfx942KfdDispatchRequestErrorV1::InvalidPointerAlignment,
            ),
            (
                Gfx942KfdDispatchPointerFixupV1::new(0, 0, 2, 4),
                Gfx942KfdDispatchRequestErrorV1::PointerTargetMisaligned,
            ),
        ];
        for (fixup, expected) in cases {
            assert_eq!(
                validate_request(
                    &request.executable_image,
                    request.descriptor_offset,
                    &request.kernarg_template,
                    request.kernarg_alignment,
                    &request.buffers,
                    &[fixup],
                    request.geometry,
                    request.private_segment_size,
                    request.group_segment_size,
                    request.timeout_milliseconds,
                ),
                Err(expected)
            );
        }
    }

    #[test]
    fn rejects_nonzero_and_duplicate_pointer_placeholders() {
        let request = valid_request().unwrap();
        let mut nonzero = request.kernarg_template.clone();
        nonzero[0] = 1;
        assert_eq!(
            validate_request(
                &request.executable_image,
                request.descriptor_offset,
                &nonzero,
                request.kernarg_alignment,
                &request.buffers,
                &request.pointer_fixups,
                request.geometry,
                request.private_segment_size,
                request.group_segment_size,
                request.timeout_milliseconds,
            ),
            Err(Gfx942KfdDispatchRequestErrorV1::NonzeroPointerPlaceholder)
        );
        assert_eq!(
            validate_request(
                &request.executable_image,
                request.descriptor_offset,
                &request.kernarg_template,
                request.kernarg_alignment,
                &request.buffers,
                &[request.pointer_fixups[0], request.pointer_fixups[0]],
                request.geometry,
                request.private_segment_size,
                request.group_segment_size,
                request.timeout_milliseconds,
            ),
            Err(Gfx942KfdDispatchRequestErrorV1::DuplicatePointerFixup)
        );
    }

    #[test]
    fn pointer_patch_writes_only_a_nonzero_address_at_each_required_alignment() {
        for (alignment, address) in [(1, 0x20_001), (2, 0x20_002), (4, 0x20_004), (8, 0x20_008)] {
            let fixup = Gfx942KfdDispatchPointerFixupV1::new(8, 0, 0, alignment);
            let mut kernarg = [0_u8; 24];
            patch_kernarg_pointer(&mut kernarg, &fixup, address).unwrap();
            assert_eq!(&kernarg[8..16], &address.to_le_bytes());
            assert_ne!(&kernarg[8..16], &[0; 8]);
        }

        let aligned = Gfx942KfdDispatchPointerFixupV1::new(0, 0, 0, 8);
        assert!(patch_kernarg_pointer(&mut [0; 8], &aligned, 0).is_err());
        assert!(patch_kernarg_pointer(&mut [0; 8], &aligned, 0x20_004).is_err());
        let invalid_alignment = Gfx942KfdDispatchPointerFixupV1::new(0, 0, 0, 0);
        assert!(patch_kernarg_pointer(&mut [0; 8], &invalid_alignment, 0x20_000).is_err());
    }

    #[test]
    fn buffer_constructor_rejects_zero_length_authority() {
        assert_eq!(
            Gfx942KfdDispatchBufferV1::new(Vec::new()),
            Err(Gfx942KfdDispatchRequestErrorV1::EmptyBuffer)
        );
    }
}
