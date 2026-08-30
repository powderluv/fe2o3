#[cfg(feature = "hardware-test-hooks")]
use crate::api::DirectRuntimeApi;
use crate::api::{ApiError, DispatchApi, QueueHandle};
use crate::environment::{AdapterCore, HsaRuntimeAdapterError, ReviewedHsaRuntimeAdapterV1};
use crate::lifecycle::{ReviewedHsaExecutableV1, ReviewedHsaKernelV1};
use fe2o3_artifacts::MAX_ABI_BYTES;
use fe2o3_host::{
    HsaDispatchObservationV1, HsaImplicitKernargInitializationObservationV1, HsaLaunchGeometryV1,
    ReviewedHsaImplicitKernargAdapterV1,
};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

#[cfg(test)]
const EXPLICIT_BYTES: usize = 48;
#[cfg(test)]
const IMPLICIT_OFFSET: usize = EXPLICIT_BYTES;
const IMPLICIT_BYTES: usize = 256;
#[cfg(test)]
const TOTAL_BYTES: usize = EXPLICIT_BYTES + IMPLICIT_BYTES;
const BLOCK_COUNT_X: usize = 0;
const BLOCK_COUNT_Y: usize = 4;
const BLOCK_COUNT_Z: usize = 8;
const GROUP_SIZE_X: usize = 12;
const GROUP_SIZE_Y: usize = 14;
const GROUP_SIZE_Z: usize = 16;
const REMAINDER_X: usize = 18;
const REMAINDER_Y: usize = 20;
const REMAINDER_Z: usize = 22;
const GLOBAL_OFFSET_X: usize = 40;
const GLOBAL_OFFSET_Y: usize = 48;
const GLOBAL_OFFSET_Z: usize = 56;
const GRID_DIMS: usize = 64;
const HOSTCALL_PTR: usize = 80;
const MULTIGRID_SYNC_ARG: usize = 88;
const HEAP_V1_PTR: usize = 96;
const DEFAULT_QUEUE_PTR: usize = 104;
const COMPLETION_ACTION: usize = 112;
const DYNAMIC_LDS_SIZE: usize = 120;
const QUEUE_PTR: usize = 200;
pub(crate) const COMPLETION_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct PendingDispatch {
    queue: QueueHandle,
    executable_identity: fe2o3_host::HsaExecutableObjectIdentityV1,
    kernel_identity: fe2o3_host::HsaKernelObjectIdentityV1,
    geometry: HsaLaunchGeometryV1,
    layout: ReviewedImplicitKernargLayout,
    kernarg_digest: [u8; 32],
}

/// Feature-gated host-visible HSA allocation used only by hardware evidence.
///
/// The allocation comes from the reviewed CPU-owned kernarg pool, is admitted
/// to the selected GPU agent, and remains live until this token is dropped.
#[cfg(feature = "hardware-test-hooks")]
pub struct ReviewedHsaHardwareTestBufferV1 {
    address: usize,
    byte_len: usize,
}

#[cfg(feature = "hardware-test-hooks")]
impl ReviewedHsaHardwareTestBufferV1 {
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn device_address(&self, byte_offset: usize) -> Result<u64, HsaRuntimeAdapterError> {
        if byte_offset >= self.byte_len {
            return Err(HsaRuntimeAdapterError::InvalidExecutableObservation(
                "hardware-test HSA buffer offset",
            ));
        }
        u64::try_from(self.address.checked_add(byte_offset).ok_or(
            HsaRuntimeAdapterError::InvalidExecutableObservation(
                "hardware-test HSA buffer address overflow",
            ),
        )?)
        .map_err(|_| {
            HsaRuntimeAdapterError::InvalidExecutableObservation(
                "hardware-test HSA buffer address conversion",
            )
        })
    }

    /// Copies host-visible bytes after the caller has synchronously quiesced
    /// every dispatch that can access this allocation.
    pub fn read_after_synchronous_dispatch(&self) -> Vec<u8> {
        // SAFETY: construction retains a live host-visible allocation of this
        // exact extent. The hardware evidence caller reads only after the
        // reviewed synchronous dispatch transition reports completion.
        unsafe { core::slice::from_raw_parts(self.address as *const u8, self.byte_len).to_vec() }
    }
}

#[cfg(feature = "hardware-test-hooks")]
impl Drop for ReviewedHsaHardwareTestBufferV1 {
    fn drop(&mut self) {
        #[cfg(fe2o3_hsa_runtime)]
        {
            // SAFETY: this token uniquely owns one live HSA pool allocation and
            // consumes it exactly once while the enclosing adapter is still live.
            let status = unsafe {
                crate::sys::fe2o3_hsa_memory_free(self.address as *mut core::ffi::c_void)
            };
            if status != crate::api::HSA_SUCCESS {
                std::process::abort();
            }
        }

        #[cfg(not(fe2o3_hsa_runtime))]
        {
            // Construction cannot succeed without the reviewed runtime. If a
            // future internal path violates that invariant, do not leak a
            // purported hardware allocation.
            std::process::abort();
        }
    }
}

/// Device timestamps for one quiesced dispatch on a profiled private queue.
#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewedHsaProfiledDispatchObservationV1 {
    start_tick: u64,
    end_tick: u64,
    timestamp_frequency_hz: u64,
    packet_id: u64,
}

#[cfg(feature = "hardware-test-hooks")]
impl ReviewedHsaProfiledDispatchObservationV1 {
    pub const fn start_tick(self) -> u64 {
        self.start_tick
    }

    pub const fn end_tick(self) -> u64 {
        self.end_tick
    }

    pub const fn timestamp_frequency_hz(self) -> u64 {
        self.timestamp_frequency_hz
    }

    pub const fn packet_id(self) -> u64 {
        self.packet_id
    }

    pub fn duration_ns(self) -> u128 {
        u128::from(self.end_tick - self.start_tick) * 1_000_000_000_u128
            / u128::from(self.timestamp_frequency_hz)
    }
}

/// Test-only persistent dispatch state used by reproducible hardware timing.
#[cfg(feature = "hardware-test-hooks")]
pub struct ReviewedHsaProfiledDispatchSessionV1<'a> {
    core: &'a mut AdapterCore<DirectRuntimeApi>,
    queue: Option<QueueHandle>,
    kernarg_address: Option<usize>,
    completion_signal: Option<u64>,
    aql_grid: [u32; 3],
    workgroup: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    timestamp_frequency_hz: u64,
}

#[cfg(feature = "hardware-test-hooks")]
impl ReviewedHsaProfiledDispatchSessionV1<'_> {
    pub fn dispatch(
        &mut self,
    ) -> Result<ReviewedHsaProfiledDispatchObservationV1, HsaRuntimeAdapterError> {
        let queue = self.queue.as_ref().expect("profiled queue remains live");
        let completion_signal = self
            .completion_signal
            .expect("profiled completion signal remains live");
        let kernarg_address = self
            .kernarg_address
            .expect("profiled kernarg allocation remains live");
        self.core
            .api
            .signal_store_release(completion_signal, 1)
            .map_err(HsaRuntimeAdapterError::api)?;
        self.core
            .api
            .queue_async_error(queue)
            .map_err(HsaRuntimeAdapterError::api)?;
        let packet_id = self
            .core
            .api
            .publish_dispatch(
                queue,
                self.aql_grid,
                self.workgroup,
                self.private_segment_size,
                self.group_segment_size,
                self.kernel_object,
                kernarg_address,
                completion_signal,
            )
            .map_err(HsaRuntimeAdapterError::api)?;
        let started = Instant::now();
        loop {
            if self.core.api.signal_load_acquire(completion_signal) == 0 {
                break;
            }
            if self.core.api.queue_async_error(queue).is_err()
                || started.elapsed() >= self.core.completion_timeout
            {
                std::process::abort();
            }
            std::thread::yield_now();
        }
        self.core
            .api
            .queue_async_error(queue)
            .map_err(HsaRuntimeAdapterError::api)?;
        let time = self
            .core
            .api
            .dispatch_time(self.core.agent, completion_signal)
            .map_err(HsaRuntimeAdapterError::api)?;
        Ok(ReviewedHsaProfiledDispatchObservationV1 {
            start_tick: time.start,
            end_tick: time.end,
            timestamp_frequency_hz: self.timestamp_frequency_hz,
            packet_id,
        })
    }
}

#[cfg(feature = "hardware-test-hooks")]
impl Drop for ReviewedHsaProfiledDispatchSessionV1<'_> {
    fn drop(&mut self) {
        if let Some(signal) = self.completion_signal.take()
            && self.core.api.signal_destroy(signal).is_err()
        {
            std::process::abort();
        }
        if let Some(address) = self.kernarg_address.take()
            && self.core.api.memory_free(address).is_err()
        {
            std::process::abort();
        }
        if let Some(mut queue) = self.queue.take()
            && self.core.api.queue_destroy(&mut queue).is_err()
        {
            std::process::abort();
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
impl ReviewedHsaRuntimeAdapterV1 {
    pub fn allocate_hardware_test_buffer(
        &mut self,
        bytes: &[u8],
    ) -> Result<ReviewedHsaHardwareTestBufferV1, HsaRuntimeAdapterError> {
        if bytes.is_empty() {
            return Err(HsaRuntimeAdapterError::InvalidExecutableObservation(
                "empty hardware-test HSA buffer",
            ));
        }
        let address = self
            .core
            .api
            .memory_allocate(self.core.kernarg_pool, bytes.len())
            .map_err(HsaRuntimeAdapterError::api)?;
        if let Err(error) = self.core.api.allow_access(self.core.agent, address) {
            if self.core.api.memory_free(address).is_err() {
                std::process::abort();
            }
            return Err(HsaRuntimeAdapterError::api(error));
        }
        self.core.api.write_memory(address, bytes);
        Ok(ReviewedHsaHardwareTestBufferV1 {
            address,
            byte_len: bytes.len(),
        })
    }

    /// Consumes a previously initialized exact dispatch binding into a
    /// persistent, profiled hardware-test session.
    ///
    /// # Safety
    ///
    /// The caller must retain the loaded executable, resolved kernel, and all
    /// allocations referenced by `kernarg` until the returned session drops.
    pub unsafe fn prepare_profiled_dispatch_session(
        &mut self,
        executable: &ReviewedHsaExecutableV1,
        kernel: &ReviewedHsaKernelV1,
        geometry: HsaLaunchGeometryV1,
        kernarg: &[u8],
    ) -> Result<ReviewedHsaProfiledDispatchSessionV1<'_>, HsaRuntimeAdapterError> {
        let mut prepared = self.pending_dispatch.take().ok_or(
            HsaRuntimeAdapterError::InvalidExecutableObservation(
                "missing reviewed profiled COV6 queue binding",
            ),
        )?;
        let state = executable.state.as_ref().ok_or_else(|| {
            reject_pending_dispatch(
                &mut self.core.api,
                &mut prepared,
                HsaRuntimeAdapterError::InvalidExecutableObservation("consumed executable"),
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(kernarg);
        let kernarg_digest: [u8; 32] = digest.finalize().into();
        if kernel.executable_identity != state.identity
            || kernarg.len() != prepared.layout.total_byte_len
            || usize::try_from(kernel.kernarg_segment_size).ok() != Some(kernarg.len())
            || kernel.kernarg_segment_alignment == 0
            || !kernel.kernarg_segment_alignment.is_power_of_two()
            || kernel.kernel_object == 0
            || prepared.executable_identity != state.identity
            || prepared.kernel_identity != kernel.identity
            || prepared.geometry != geometry
            || prepared.kernarg_digest != kernarg_digest
        {
            return Err(reject_pending_dispatch(
                &mut self.core.api,
                &mut prepared,
                HsaRuntimeAdapterError::InvalidExecutableObservation(
                    "profiled dispatch handle, geometry, or kernarg binding",
                ),
            ));
        }
        let aql_grid = match checked_aql_grid(geometry) {
            Ok(grid) => grid,
            Err(error) => {
                return Err(reject_pending_dispatch(
                    &mut self.core.api,
                    &mut prepared,
                    error,
                ));
            }
        };
        let group_segment_size = match kernel
            .group_segment_size
            .checked_add(geometry.dynamic_shared_memory_bytes())
        {
            Some(size) => size,
            None => {
                return Err(reject_pending_dispatch(
                    &mut self.core.api,
                    &mut prepared,
                    HsaRuntimeAdapterError::InvalidExecutableObservation(
                        "profiled group segment size overflow",
                    ),
                ));
            }
        };
        if let Err(primary) = self.core.api.queue_enable_profiling(&prepared.queue) {
            return Err(cleanup_dispatch(
                &mut self.core.api,
                None,
                Some(prepared.queue),
                None,
                primary,
            ));
        }
        let address = match self
            .core
            .api
            .memory_allocate(self.core.kernarg_pool, kernarg.len())
        {
            Ok(address) => address,
            Err(primary) => {
                return Err(cleanup_dispatch(
                    &mut self.core.api,
                    None,
                    Some(prepared.queue),
                    None,
                    primary,
                ));
            }
        };
        if !address.is_multiple_of(kernel.kernarg_segment_alignment as usize) {
            return Err(cleanup_dispatch(
                &mut self.core.api,
                Some(address),
                Some(prepared.queue),
                None,
                ApiError {
                    operation: "validate profiled HSA kernarg alignment",
                    status: -1,
                },
            ));
        }
        if let Err(primary) = self.core.api.allow_access(self.core.agent, address) {
            return Err(cleanup_dispatch(
                &mut self.core.api,
                Some(address),
                Some(prepared.queue),
                None,
                primary,
            ));
        }
        self.core.api.write_memory(address, kernarg);
        let signal = match self.core.api.signal_create(1) {
            Ok(signal) => signal,
            Err(primary) => {
                return Err(cleanup_dispatch(
                    &mut self.core.api,
                    Some(address),
                    Some(prepared.queue),
                    None,
                    primary,
                ));
            }
        };
        let timestamp_frequency_hz = match self.core.api.timestamp_frequency() {
            Ok(frequency) => frequency,
            Err(primary) => {
                return Err(cleanup_dispatch(
                    &mut self.core.api,
                    Some(address),
                    Some(prepared.queue),
                    Some(signal),
                    primary,
                ));
            }
        };
        Ok(ReviewedHsaProfiledDispatchSessionV1 {
            core: &mut self.core,
            queue: Some(prepared.queue),
            kernarg_address: Some(address),
            completion_signal: Some(signal),
            aql_grid,
            workgroup: geometry.workgroup(),
            private_segment_size: kernel.private_segment_size,
            group_segment_size,
            kernel_object: kernel.kernel_object,
            timestamp_frequency_hz,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewedImplicitKernargLayout {
    explicit_byte_len: usize,
    implicit_byte_offset: usize,
    implicit_byte_len: usize,
    total_byte_len: usize,
}

struct PreSubmitDispatch {
    queue: QueueHandle,
    kernarg_address: usize,
    completion_signal: u64,
}

struct SubmittedDispatch {
    resources: PreSubmitDispatch,
    packet_id: u64,
}

struct QuiescedDispatch(SubmittedDispatch);

struct UnquiescedDispatch {
    submitted: SubmittedDispatch,
    reason: UnquiescedReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnquiescedReason {
    QueueError(ApiError),
    CompletionDeadline {
        last_observation: i64,
    },
    #[cfg(feature = "hardware-test-hooks")]
    TestPhaseEvidence,
}

enum CompletionTransition {
    Quiesced {
        dispatch: QuiescedDispatch,
        queue_error: Option<ApiError>,
    },
    Unquiesced(UnquiescedDispatch),
}

// SAFETY: the bounded explicit prefix is supplied by the reviewed host
// lifecycle and preserved byte-for-byte. An exact 256-byte COV6 hidden span is
// initialized from reviewed geometry and the exact private HSA queue retained
// for the following launch. An explicit-only COV6 ABI still binds that exact
// queue but performs no kernarg writes. Every other layout and launch
// substitution is rejected.
unsafe impl ReviewedHsaImplicitKernargAdapterV1 for ReviewedHsaRuntimeAdapterV1 {
    unsafe fn initialize_implicit_kernarg(
        &mut self,
        executable: &Self::Executable,
        kernel: &Self::Kernel,
        geometry: HsaLaunchGeometryV1,
        explicit_byte_len: usize,
        implicit_byte_offset: usize,
        implicit_byte_len: usize,
        kernarg: &mut [u8],
    ) -> Result<HsaImplicitKernargInitializationObservationV1, Self::Error> {
        prepare_implicit_kernarg(
            &mut self.core,
            &mut self.pending_dispatch,
            executable,
            kernel,
            geometry,
            explicit_byte_len,
            implicit_byte_offset,
            implicit_byte_len,
            kernarg,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_implicit_kernarg(
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    geometry: HsaLaunchGeometryV1,
    explicit_byte_len: usize,
    implicit_byte_offset: usize,
    implicit_byte_len: usize,
    kernarg: &[u8],
) -> Result<ReviewedImplicitKernargLayout, HsaRuntimeAdapterError> {
    let executable =
        executable
            .state
            .as_ref()
            .ok_or(HsaRuntimeAdapterError::InvalidImplicitKernarg(
                "consumed executable",
            ))?;
    if kernel.executable_identity != executable.identity {
        return Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
            "kernel/executable identity",
        ));
    }
    let explicit_byte_len_u64 = u64::try_from(explicit_byte_len).map_err(|_| {
        HsaRuntimeAdapterError::InvalidImplicitKernarg("bounded explicit kernarg prefix")
    })?;
    let total_byte_len = explicit_byte_len.checked_add(implicit_byte_len).ok_or(
        HsaRuntimeAdapterError::InvalidImplicitKernarg("kernarg layout size overflow"),
    )?;
    if explicit_byte_len_u64 > MAX_ABI_BYTES
        || implicit_byte_offset != explicit_byte_len
        || !matches!(implicit_byte_len, 0 | IMPLICIT_BYTES)
        || kernarg.len() != total_byte_len
        || usize::try_from(kernel.kernarg_segment_size).ok() != Some(kernarg.len())
        || kernel.kernarg_segment_alignment == 0
        || !kernel.kernarg_segment_alignment.is_power_of_two()
    {
        return Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
            "bounded explicit prefix plus zero or exact 256-byte hidden span",
        ));
    }
    let grid = geometry.grid();
    let workgroup = geometry.workgroup();
    if grid.contains(&0)
        || workgroup.contains(&0)
        || workgroup.iter().any(|value| u16::try_from(*value).is_err())
    {
        return Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
            "launch geometry",
        ));
    }
    kernel
        .group_segment_size
        .checked_add(geometry.dynamic_shared_memory_bytes())
        .ok_or(HsaRuntimeAdapterError::InvalidImplicitKernarg(
            "AQL group segment size overflow",
        ))?;
    Ok(ReviewedImplicitKernargLayout {
        explicit_byte_len,
        implicit_byte_offset,
        implicit_byte_len,
        total_byte_len,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_implicit_kernarg<A: DispatchApi>(
    core: &mut AdapterCore<A>,
    pending: &mut Option<PendingDispatch>,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    geometry: HsaLaunchGeometryV1,
    explicit_byte_len: usize,
    implicit_byte_offset: usize,
    implicit_byte_len: usize,
    kernarg: &mut [u8],
) -> Result<HsaImplicitKernargInitializationObservationV1, HsaRuntimeAdapterError> {
    if pending.is_some() {
        return Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
            "one unconsumed queue binding already exists",
        ));
    }
    let layout = validate_implicit_kernarg(
        executable,
        kernel,
        geometry,
        explicit_byte_len,
        implicit_byte_offset,
        implicit_byte_len,
        kernarg,
    )?;
    let state = executable
        .state
        .as_ref()
        .expect("validated executable remains live");
    let queue_size = reviewed_queue_size(core.queue_min_size, core.queue_max_size)?;
    let mut queue = core
        .api
        .queue_create(core.agent, queue_size)
        .map_err(HsaRuntimeAdapterError::api)?;
    if let Err(primary) = core.api.queue_async_error(&queue) {
        if core.api.queue_destroy(&mut queue).is_err() {
            std::process::abort();
        }
        return Err(HsaRuntimeAdapterError::api(primary));
    }
    let queue_pointer = match u64::try_from(queue.pointer()) {
        Ok(pointer) if pointer != 0 => pointer,
        _ => {
            let primary = ApiError {
                operation: "validate HSA queue pointer for COV6 kernarg",
                status: -1,
            };
            if core.api.queue_destroy(&mut queue).is_err() {
                std::process::abort();
            }
            return Err(HsaRuntimeAdapterError::api(primary));
        }
    };

    let grid = geometry.grid();
    let workgroup = geometry.workgroup();
    let implicit_offset = layout.implicit_byte_offset;
    let explicit = kernarg[..layout.explicit_byte_len].to_vec();
    if layout.implicit_byte_len == IMPLICIT_BYTES {
        kernarg[implicit_offset..layout.total_byte_len].fill(0);
        put_u32(kernarg, implicit_offset + BLOCK_COUNT_X, grid[0]);
        put_u32(kernarg, implicit_offset + BLOCK_COUNT_Y, grid[1]);
        put_u32(kernarg, implicit_offset + BLOCK_COUNT_Z, grid[2]);
        put_u16(kernarg, implicit_offset + GROUP_SIZE_X, workgroup[0] as u16);
        put_u16(kernarg, implicit_offset + GROUP_SIZE_Y, workgroup[1] as u16);
        put_u16(kernarg, implicit_offset + GROUP_SIZE_Z, workgroup[2] as u16);
        put_u16(kernarg, implicit_offset + REMAINDER_X, 0);
        put_u16(kernarg, implicit_offset + REMAINDER_Y, 0);
        put_u16(kernarg, implicit_offset + REMAINDER_Z, 0);
        put_u64(kernarg, implicit_offset + GLOBAL_OFFSET_X, 0);
        put_u64(kernarg, implicit_offset + GLOBAL_OFFSET_Y, 0);
        put_u64(kernarg, implicit_offset + GLOBAL_OFFSET_Z, 0);
        let dimensions = if grid[2]
            .checked_mul(workgroup[2])
            .is_some_and(|size| size > 1)
        {
            3
        } else if grid[1]
            .checked_mul(workgroup[1])
            .is_some_and(|size| size > 1)
        {
            2
        } else {
            1
        };
        put_u16(kernarg, implicit_offset + GRID_DIMS, dimensions);
        put_u64(kernarg, implicit_offset + HOSTCALL_PTR, 0);
        put_u64(kernarg, implicit_offset + MULTIGRID_SYNC_ARG, 0);
        put_u64(kernarg, implicit_offset + HEAP_V1_PTR, 0);
        put_u64(kernarg, implicit_offset + DEFAULT_QUEUE_PTR, 0);
        put_u64(kernarg, implicit_offset + COMPLETION_ACTION, 0);
        put_u32(
            kernarg,
            implicit_offset + DYNAMIC_LDS_SIZE,
            geometry.dynamic_shared_memory_bytes(),
        );
        put_u64(kernarg, implicit_offset + QUEUE_PTR, queue_pointer);
    }
    if kernarg[..layout.explicit_byte_len] != explicit {
        std::process::abort();
    }
    let mut digest = Sha256::new();
    digest.update(&*kernarg);
    let kernarg_digest = digest.finalize().into();
    *pending = Some(PendingDispatch {
        queue,
        executable_identity: state.identity,
        kernel_identity: kernel.identity,
        geometry,
        layout,
        kernarg_digest,
    });
    let explicit_byte_len =
        u64::try_from(layout.explicit_byte_len).expect("validated explicit length fits u64");
    let implicit_byte_offset =
        u64::try_from(layout.implicit_byte_offset).expect("validated implicit offset fits u64");
    let implicit_byte_len =
        u64::try_from(layout.implicit_byte_len).expect("validated implicit length fits u64");
    Ok(HsaImplicitKernargInitializationObservationV1::new(
        state.identity,
        kernel.identity,
        geometry,
        explicit_byte_len,
        implicit_byte_offset,
        implicit_byte_len,
        true,
    ))
}

pub(crate) fn launch_and_wait<A: DispatchApi>(
    core: &mut AdapterCore<A>,
    pending: &mut Option<PendingDispatch>,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    geometry: HsaLaunchGeometryV1,
    kernarg: &mut [u8],
) -> Result<HsaDispatchObservationV1, HsaRuntimeAdapterError> {
    let mut prepared =
        pending
            .take()
            .ok_or(HsaRuntimeAdapterError::InvalidExecutableObservation(
                "missing reviewed COV6 queue binding",
            ))?;
    let executable = match executable.state.as_ref() {
        Some(executable) => executable,
        None => {
            return Err(reject_pending_dispatch(
                &mut core.api,
                &mut prepared,
                HsaRuntimeAdapterError::InvalidExecutableObservation("consumed executable"),
            ));
        }
    };
    let mut digest = Sha256::new();
    digest.update(&*kernarg);
    let kernarg_digest: [u8; 32] = digest.finalize().into();
    if kernel.executable_identity != executable.identity
        || kernarg.len() != prepared.layout.total_byte_len
        || usize::try_from(kernel.kernarg_segment_size).ok() != Some(kernarg.len())
        || kernel.kernarg_segment_alignment == 0
        || !kernel.kernarg_segment_alignment.is_power_of_two()
        || kernel.kernel_object == 0
        || kernel.symbol == 0
        || prepared.executable_identity != executable.identity
        || prepared.kernel_identity != kernel.identity
        || prepared.geometry != geometry
        || prepared.kernarg_digest != kernarg_digest
    {
        return Err(reject_pending_dispatch(
            &mut core.api,
            &mut prepared,
            HsaRuntimeAdapterError::InvalidExecutableObservation(
                "dispatch handle, geometry, or kernarg binding",
            ),
        ));
    }
    let aql_grid = match checked_aql_grid(geometry) {
        Ok(grid) => grid,
        Err(error) => {
            return Err(reject_pending_dispatch(&mut core.api, &mut prepared, error));
        }
    };
    let group_segment_size = match kernel
        .group_segment_size
        .checked_add(geometry.dynamic_shared_memory_bytes())
    {
        Some(size) => size,
        None => {
            return Err(reject_pending_dispatch(
                &mut core.api,
                &mut prepared,
                HsaRuntimeAdapterError::InvalidExecutableObservation("group segment size overflow"),
            ));
        }
    };
    let generation = core.next_identity;
    let next_generation = match generation.checked_add(1) {
        Some(next) => next,
        None => {
            return Err(reject_pending_dispatch(
                &mut core.api,
                &mut prepared,
                HsaRuntimeAdapterError::InvalidExecutableObservation(
                    "dispatch generation overflow",
                ),
            ));
        }
    };

    let address = match core.api.memory_allocate(core.kernarg_pool, kernarg.len()) {
        Ok(address) => address,
        Err(primary) => {
            return Err(cleanup_dispatch(
                &mut core.api,
                None,
                Some(prepared.queue),
                None,
                primary,
            ));
        }
    };
    let required_alignment = kernel.kernarg_segment_alignment as usize;
    if !address.is_multiple_of(required_alignment) {
        return Err(cleanup_dispatch(
            &mut core.api,
            Some(address),
            Some(prepared.queue),
            None,
            ApiError {
                operation: "validate HSA kernarg allocation alignment",
                status: -1,
            },
        ));
    }
    if let Err(primary) = core.api.allow_access(core.agent, address) {
        return Err(cleanup_dispatch(
            &mut core.api,
            Some(address),
            Some(prepared.queue),
            None,
            primary,
        ));
    }
    core.api.write_memory(address, kernarg);
    let queue = prepared.queue;
    let signal = match core.api.signal_create(1) {
        Ok(signal) => signal,
        Err(primary) => {
            return Err(cleanup_dispatch(
                &mut core.api,
                Some(address),
                Some(queue),
                None,
                primary,
            ));
        }
    };
    if let Err(primary) = core.api.queue_async_error(&queue) {
        return Err(cleanup_dispatch(
            &mut core.api,
            Some(address),
            Some(queue),
            Some(signal),
            primary,
        ));
    }
    let pre_submit = PreSubmitDispatch {
        queue,
        kernarg_address: address,
        completion_signal: signal,
    };
    let packet_id = match core.api.publish_dispatch(
        &pre_submit.queue,
        aql_grid,
        geometry.workgroup(),
        kernel.private_segment_size,
        group_segment_size,
        kernel.kernel_object,
        pre_submit.kernarg_address,
        pre_submit.completion_signal,
    ) {
        Ok(packet_id) => packet_id,
        Err(primary) => {
            return Err(cleanup_dispatch(
                &mut core.api,
                Some(pre_submit.kernarg_address),
                Some(pre_submit.queue),
                Some(pre_submit.completion_signal),
                primary,
            ));
        }
    };
    core.next_identity = next_generation;
    let submitted = SubmittedDispatch {
        resources: pre_submit,
        packet_id,
    };
    let (quiesced, queue_error) =
        match await_quiescence(&mut core.api, submitted, core.completion_timeout) {
            CompletionTransition::Quiesced {
                dispatch,
                queue_error,
            } => (dispatch, queue_error),
            CompletionTransition::Unquiesced(unquiesced) => terminate_unquiesced(unquiesced),
        };
    let SubmittedDispatch {
        resources,
        packet_id,
    } = quiesced.0;
    let queue_id = resources.queue.id();
    let signal = resources.completion_signal;
    match queue_error {
        Some(primary) => {
            return Err(cleanup_dispatch(
                &mut core.api,
                Some(resources.kernarg_address),
                Some(resources.queue),
                Some(resources.completion_signal),
                primary,
            ));
        }
        None => cleanup_completed(
            &mut core.api,
            resources.kernarg_address,
            resources.queue,
            resources.completion_signal,
        ),
    }
    let dispatch_identity = derive_dispatch_identity(
        core.environment.runtime().instance(),
        queue_id,
        packet_id,
        signal,
        generation,
    );
    HsaDispatchObservationV1::new(
        dispatch_identity,
        executable.identity,
        kernel.identity,
        geometry,
        true,
    )
    .map_err(|_| HsaRuntimeAdapterError::InvalidExecutableObservation("dispatch identity"))
}

fn await_quiescence<A: DispatchApi>(
    api: &mut A,
    submitted: SubmittedDispatch,
    timeout: Duration,
) -> CompletionTransition {
    #[cfg(feature = "hardware-test-hooks")]
    if record_post_submit_wait_phase().is_err() {
        return CompletionTransition::Unquiesced(UnquiescedDispatch {
            submitted,
            reason: UnquiescedReason::TestPhaseEvidence,
        });
    }
    let started = Instant::now();
    let deadline = started.checked_add(timeout).unwrap_or(started);
    let last_observation = loop {
        let observation = api.signal_load_acquire(submitted.resources.completion_signal);
        if observation == 0 {
            let queue_error = api.queue_async_error(&submitted.resources.queue).err();
            return CompletionTransition::Quiesced {
                dispatch: QuiescedDispatch(submitted),
                queue_error,
            };
        }
        if let Err(error) = api.queue_async_error(&submitted.resources.queue) {
            return CompletionTransition::Unquiesced(UnquiescedDispatch {
                submitted,
                reason: UnquiescedReason::QueueError(error),
            });
        }
        if Instant::now() >= deadline {
            break observation;
        }
        std::thread::yield_now();
    };
    CompletionTransition::Unquiesced(UnquiescedDispatch {
        submitted,
        reason: UnquiescedReason::CompletionDeadline { last_observation },
    })
}

#[cfg(feature = "hardware-test-hooks")]
fn record_post_submit_wait_phase() -> std::io::Result<()> {
    use std::io::Write;

    const VARIABLE: &str = "FE2O3_HSA_TEST_POST_SUBMIT_PHASE";
    const RECORD: &[u8] = b"fe2o3-hsa-post-submit-wait-v1\n";
    let Some(path) = std::env::var_os(VARIABLE) else {
        return Ok(());
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(RECORD)?;
    file.sync_all()
}

fn terminate_unquiesced(unquiesced: UnquiescedDispatch) -> ! {
    let _reason = unquiesced.reason;
    let _packet_id = unquiesced.submitted.packet_id;
    #[cfg(feature = "hardware-test-hooks")]
    let _ = record_test_terminal_reason(unquiesced.reason);
    // Returning would release caller-side allocations while the GPU may still
    // reference them. Process termination is the production terminal policy.
    let _retained_authority = std::mem::ManuallyDrop::new(unquiesced);
    std::process::abort()
}

#[cfg(feature = "hardware-test-hooks")]
fn record_test_terminal_reason(reason: UnquiescedReason) -> std::io::Result<()> {
    use std::io::Write;

    const VARIABLE: &str = "FE2O3_HSA_TEST_POST_SUBMIT_PHASE";
    let Some(path) = std::env::var_os(VARIABLE) else {
        return Ok(());
    };
    let record: &[u8] = match reason {
        UnquiescedReason::QueueError(_) => b"fe2o3-hsa-unquiesced-queue-error-v1\n",
        UnquiescedReason::CompletionDeadline { .. } => b"fe2o3-hsa-unquiesced-deadline-v1\n",
        UnquiescedReason::TestPhaseEvidence => b"fe2o3-hsa-test-evidence-failure-v1\n",
    };
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    file.write_all(record)?;
    file.sync_all()
}

fn checked_aql_grid(geometry: HsaLaunchGeometryV1) -> Result<[u32; 3], HsaRuntimeAdapterError> {
    let blocks = geometry.grid();
    let workgroup = geometry.workgroup();
    let mut result = [0; 3];
    for index in 0..3 {
        result[index] = blocks[index].checked_mul(workgroup[index]).ok_or(
            HsaRuntimeAdapterError::InvalidExecutableObservation("AQL grid size overflow"),
        )?;
        if result[index] == 0 {
            return Err(HsaRuntimeAdapterError::InvalidExecutableObservation(
                "zero AQL grid size",
            ));
        }
    }
    Ok(result)
}

fn reviewed_queue_size(minimum: u32, maximum: u32) -> Result<u32, HsaRuntimeAdapterError> {
    if minimum == 0 || maximum < minimum || !minimum.is_power_of_two() || !maximum.is_power_of_two()
    {
        return Err(HsaRuntimeAdapterError::InvalidExecutableObservation(
            "HSA queue limits",
        ));
    }
    Ok(64_u32.clamp(minimum, maximum))
}

pub(crate) fn destroy_pending_dispatch<A: DispatchApi>(
    api: &mut A,
    pending: &mut Option<PendingDispatch>,
) {
    if let Some(mut pending) = pending.take()
        && api.queue_destroy(&mut pending.queue).is_err()
    {
        std::process::abort();
    }
}

fn reject_pending_dispatch<A: DispatchApi>(
    api: &mut A,
    pending: &mut PendingDispatch,
    primary: HsaRuntimeAdapterError,
) -> HsaRuntimeAdapterError {
    if api.queue_destroy(&mut pending.queue).is_err() {
        std::process::abort();
    }
    primary
}

fn cleanup_dispatch<A: DispatchApi>(
    api: &mut A,
    address: Option<usize>,
    mut queue: Option<QueueHandle>,
    signal: Option<u64>,
    primary: ApiError,
) -> HsaRuntimeAdapterError {
    if let Some(signal) = signal
        && api.signal_destroy(signal).is_err()
    {
        std::process::abort();
    }
    if let Some(address) = address
        && api.memory_free(address).is_err()
    {
        std::process::abort();
    }
    if let Some(queue) = queue.as_mut()
        && api.queue_destroy(queue).is_err()
    {
        std::process::abort();
    }
    HsaRuntimeAdapterError::api(primary)
}

fn cleanup_completed<A: DispatchApi>(
    api: &mut A,
    address: usize,
    mut queue: QueueHandle,
    signal: u64,
) {
    if api.signal_destroy(signal).is_err() {
        std::process::abort();
    }
    if api.memory_free(address).is_err() {
        std::process::abort();
    }
    if api.queue_destroy(&mut queue).is_err() {
        std::process::abort();
    }
}

fn derive_dispatch_identity(
    runtime: [u8; 16],
    queue: u64,
    packet: u64,
    signal: u64,
    generation: u64,
) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"fe2o3-hsa-aql-dispatch-v1\0");
    hasher.update(runtime);
    hasher.update(queue.to_le_bytes());
    hasher.update(packet.to_le_bytes());
    hasher.update(signal.to_le_bytes());
    hasher.update(generation.to_le_bytes());
    let digest = hasher.finalize();
    let mut result = [0; 16];
    result.copy_from_slice(&digest[..16]);
    result
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        AgentFacts, EnvironmentApi, ExecutableApi, HipFacts, PoolFacts, RuntimeFacts, SymbolFacts,
    };
    use crate::lifecycle::ExecutableState;
    use fe2o3_amd_target::AmdTargetId;
    use fe2o3_artifacts::{DigestAlgorithm, PayloadDigest};
    use fe2o3_host::{
        HsaAgentIdentityV1, HsaEnvironmentObservationV1, HsaExecutableObjectIdentityV1,
        HsaKernelObjectIdentityV1, HsaPhysicalDeviceIdentityV1, HsaRuntimeIdentityV1,
    };
    use std::collections::{BTreeMap, VecDeque};

    #[cfg(feature = "hardware-test-hooks")]
    #[test]
    fn hardware_test_buffer_addresses_are_extent_checked() {
        let buffer = core::mem::ManuallyDrop::new(ReviewedHsaHardwareTestBufferV1 {
            address: 0x1000,
            byte_len: 16,
        });
        assert_eq!(buffer.byte_len(), 16);
        assert_eq!(buffer.device_address(12).unwrap(), 0x100c);
        assert!(buffer.device_address(16).is_err());

        let overflowing = core::mem::ManuallyDrop::new(ReviewedHsaHardwareTestBufferV1 {
            address: usize::MAX - 1,
            byte_len: 8,
        });
        assert!(overflowing.device_address(4).is_err());
    }

    #[derive(Default)]
    struct MockApi {
        log: Vec<&'static str>,
        failures: BTreeMap<&'static str, i32>,
        async_calls: usize,
        fail_async_call: Option<usize>,
        completion: i64,
        completion_sequence: VecDeque<i64>,
        written: Vec<u8>,
        published_grid: Option<[u32; 3]>,
        published_group_segment_bytes: Option<u32>,
    }

    impl MockApi {
        fn call(&mut self, operation: &'static str) -> Result<(), ApiError> {
            self.log.push(operation);
            match self.failures.get(operation) {
                Some(status) => Err(ApiError {
                    operation,
                    status: *status,
                }),
                None => Ok(()),
            }
        }
    }

    impl EnvironmentApi for MockApi {
        fn initialize(&mut self) -> Result<RuntimeFacts, ApiError> {
            unreachable!()
        }

        fn shut_down(&mut self) -> Result<(), ApiError> {
            self.call("shutdown")
        }

        fn observe_hip_device(&mut self, _ordinal: i32) -> Result<HipFacts, ApiError> {
            unreachable!()
        }

        fn collect_agents(&mut self) -> Result<Vec<AgentFacts>, ApiError> {
            unreachable!()
        }

        fn collect_kernarg_pools(&mut self) -> Result<Vec<PoolFacts>, ApiError> {
            unreachable!()
        }
    }

    impl ExecutableApi for MockApi {
        fn reader_create(&mut self, _bytes: &[u8]) -> Result<u64, ApiError> {
            unreachable!()
        }

        fn reader_destroy(&mut self, _reader: u64) -> Result<(), ApiError> {
            self.call("reader_destroy")
        }

        fn executable_create(&mut self, _profile: u32) -> Result<u64, ApiError> {
            unreachable!()
        }

        fn executable_load(
            &mut self,
            _executable: u64,
            _agent: u64,
            _reader: u64,
        ) -> Result<u64, ApiError> {
            unreachable!()
        }

        fn executable_freeze(&mut self, _executable: u64) -> Result<(), ApiError> {
            unreachable!()
        }

        fn executable_destroy(&mut self, _executable: u64) -> Result<(), ApiError> {
            self.call("executable_destroy")
        }

        fn resolve_symbol(
            &mut self,
            _executable: u64,
            _agent: u64,
            _name: &str,
        ) -> Result<SymbolFacts, ApiError> {
            unreachable!()
        }
    }

    impl DispatchApi for MockApi {
        fn memory_allocate(&mut self, _pool: u64, _len: usize) -> Result<usize, ApiError> {
            self.call("memory_allocate")?;
            Ok(0x1000)
        }

        fn allow_access(&mut self, _agent: u64, _address: usize) -> Result<(), ApiError> {
            self.call("allow_access")
        }

        fn write_memory(&mut self, _address: usize, bytes: &[u8]) {
            self.log.push("write_memory");
            self.written = bytes.to_vec();
        }

        fn memory_free(&mut self, _address: usize) -> Result<(), ApiError> {
            self.call("memory_free")
        }

        fn queue_create(&mut self, _agent: u64, size: u32) -> Result<QueueHandle, ApiError> {
            self.call("queue_create")?;
            Ok(QueueHandle::for_test(0xabc0, 41, size))
        }

        fn queue_async_error(&mut self, _queue: &QueueHandle) -> Result<(), ApiError> {
            self.log.push("queue_async_error");
            self.async_calls += 1;
            if self.fail_async_call == Some(self.async_calls) {
                Err(ApiError {
                    operation: "queue_async_error",
                    status: 82,
                })
            } else {
                self.failures
                    .get("queue_async_error")
                    .map_or(Ok(()), |status| {
                        Err(ApiError {
                            operation: "queue_async_error",
                            status: *status,
                        })
                    })
            }
        }

        fn queue_destroy(&mut self, _queue: &mut QueueHandle) -> Result<(), ApiError> {
            self.call("queue_destroy")
        }

        fn signal_create(&mut self, _initial_value: i64) -> Result<u64, ApiError> {
            self.call("signal_create")?;
            Ok(51)
        }

        fn signal_destroy(&mut self, _signal: u64) -> Result<(), ApiError> {
            self.call("signal_destroy")
        }

        fn signal_load_acquire(&mut self, _signal: u64) -> i64 {
            self.log.push("signal_load");
            self.completion_sequence
                .pop_front()
                .unwrap_or(self.completion)
        }

        fn publish_dispatch(
            &mut self,
            _queue: &QueueHandle,
            grid: [u32; 3],
            _workgroup: [u32; 3],
            _private_segment_size: u32,
            group_segment_size: u32,
            _kernel_object: u64,
            _kernarg: usize,
            _completion_signal: u64,
        ) -> Result<u64, ApiError> {
            self.call("publish")?;
            self.published_grid = Some(grid);
            self.published_group_segment_bytes = Some(group_segment_size);
            Ok(61)
        }
    }

    fn environment() -> HsaEnvironmentObservationV1 {
        let target = AmdTargetId::parse("gfx942").unwrap();
        let runtime = HsaRuntimeIdentityV1::new(
            "ROCr",
            "1.18",
            DigestAlgorithm::Sha256.calculate(b"runtime"),
            [1; 16],
        )
        .unwrap();
        let physical = HsaPhysicalDeviceIdentityV1::new([2; 16], 2, 0, target).unwrap();
        let agent = HsaAgentIdentityV1::new([1; 16], 20, [2; 16], target).unwrap();
        HsaEnvironmentObservationV1::new(runtime, physical, agent).unwrap()
    }

    fn make_core(api: MockApi) -> AdapterCore<MockApi> {
        AdapterCore {
            api,
            environment: environment(),
            agent: 20,
            profile: 0,
            queue_min_size: 64,
            queue_max_size: 1024,
            kernarg_pool: 30,
            completion_timeout: COMPLETION_TIMEOUT,
            next_identity: 1,
            runtime_live: true,
            _context: None,
        }
    }

    fn handles() -> (ReviewedHsaExecutableV1, ReviewedHsaKernelV1) {
        let executable_identity = HsaExecutableObjectIdentityV1::new([3; 32]).unwrap();
        let kernel_identity = HsaKernelObjectIdentityV1::new([4; 32]).unwrap();
        (
            ReviewedHsaExecutableV1 {
                state: Some(ExecutableState {
                    bytes: b"code".to_vec().into_boxed_slice(),
                    reader: 11,
                    executable: 12,
                    _loaded_code_object: 13,
                    identity: executable_identity,
                }),
            },
            ReviewedHsaKernelV1 {
                symbol: 14,
                kernel_object: 15,
                executable_identity,
                identity: kernel_identity,
                kernarg_segment_size: TOTAL_BYTES as u32,
                kernarg_segment_alignment: 8,
                group_segment_size: 32,
                private_segment_size: 64,
            },
        )
    }

    fn geometry() -> HsaLaunchGeometryV1 {
        HsaLaunchGeometryV1::new([2, 1, 1], [256, 1, 1], 0)
    }

    fn kernarg() -> [u8; TOTAL_BYTES] {
        let mut bytes = [0; TOTAL_BYTES];
        for (index, byte) in bytes[..EXPLICIT_BYTES].iter_mut().enumerate() {
            *byte = index as u8;
        }
        bytes
    }

    fn handles_for_explicit_prefix(
        explicit_byte_len: usize,
    ) -> (ReviewedHsaExecutableV1, ReviewedHsaKernelV1) {
        let (executable, mut kernel) = handles();
        kernel.kernarg_segment_size = u32::try_from(
            explicit_byte_len
                .checked_add(IMPLICIT_BYTES)
                .expect("test kernarg size must not overflow"),
        )
        .expect("test kernarg size must fit u32");
        (executable, kernel)
    }

    fn kernarg_for_explicit_prefix(explicit_byte_len: usize) -> Vec<u8> {
        let mut bytes = vec![0; explicit_byte_len + IMPLICIT_BYTES];
        for (index, byte) in bytes[..explicit_byte_len].iter_mut().enumerate() {
            *byte = index.wrapping_mul(17).wrapping_add(3) as u8;
        }
        bytes
    }

    #[test]
    fn worker_v3_geometry_binds_hidden_and_aql_dynamic_lds() {
        for dynamic_lds in [256, 0] {
            let (executable, mut kernel) = handles_for_explicit_prefix(40);
            kernel.group_segment_size = 32;
            kernel.kernarg_segment_alignment = 16;
            let geometry = HsaLaunchGeometryV1::new([1, 1, 1], [64, 1, 1], dynamic_lds);
            let mut bytes = kernarg_for_explicit_prefix(40);
            let explicit = bytes[..40].to_vec();
            let mut core = make_core(MockApi::default());
            let mut pending = None;

            let observation = prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry,
                40,
                40,
                256,
                &mut bytes,
            )
            .unwrap();
            assert_eq!(observation.geometry(), geometry);
            assert_eq!(&bytes[..40], explicit);
            assert_eq!(
                u32::from_le_bytes(bytes[160..164].try_into().unwrap()),
                dynamic_lds
            );

            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry,
                &mut bytes,
            )
            .unwrap();
            assert_eq!(
                core.api.published_group_segment_bytes,
                Some(32 + dynamic_lds)
            );
        }
    }

    #[test]
    fn dynamic_lds_group_segment_overflow_fails_before_queue() {
        let (executable, mut kernel) = handles_for_explicit_prefix(40);
        kernel.group_segment_size = u32::MAX;
        let mut bytes = kernarg_for_explicit_prefix(40);
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let geometry = HsaLaunchGeometryV1::new([1, 1, 1], [64, 1, 1], 1);
        assert!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry,
                40,
                40,
                256,
                &mut bytes,
            )
            .is_err()
        );
        assert!(core.api.log.is_empty());
        assert!(pending.is_none());
    }

    #[test]
    fn cov6_hidden_layout_is_exact_for_variable_explicit_prefixes() {
        for explicit_byte_len in [16, EXPLICIT_BYTES, 80] {
            let (executable, kernel) = handles_for_explicit_prefix(explicit_byte_len);
            let mut core = make_core(MockApi::default());
            let mut pending = None;
            let mut bytes = kernarg_for_explicit_prefix(explicit_byte_len);
            let explicit = bytes[..explicit_byte_len].to_vec();
            let observation = prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                explicit_byte_len,
                explicit_byte_len,
                IMPLICIT_BYTES,
                &mut bytes,
            )
            .unwrap();
            assert!(observation.initialized());
            assert_eq!(&bytes[..explicit_byte_len], explicit);
            let mut expected = [0_u8; IMPLICIT_BYTES];
            expected[0..4].copy_from_slice(&2_u32.to_le_bytes());
            expected[4..8].copy_from_slice(&1_u32.to_le_bytes());
            expected[8..12].copy_from_slice(&1_u32.to_le_bytes());
            expected[12..14].copy_from_slice(&256_u16.to_le_bytes());
            expected[14..16].copy_from_slice(&1_u16.to_le_bytes());
            expected[16..18].copy_from_slice(&1_u16.to_le_bytes());
            expected[64..66].copy_from_slice(&1_u16.to_le_bytes());
            expected[200..208].copy_from_slice(&0xabc0_u64.to_le_bytes());
            assert_eq!(&bytes[explicit_byte_len..], expected);
            destroy_pending_dispatch(&mut core.api, &mut pending);
        }
    }

    #[test]
    fn explicit_only_layout_preserves_kernarg_and_owns_one_queue_through_launch() {
        let (executable, mut kernel) = handles();
        kernel.kernarg_segment_size = EXPLICIT_BYTES as u32;
        let mut bytes = kernarg()[..EXPLICIT_BYTES].to_vec();
        let original = bytes.clone();
        let mut core = make_core(MockApi::default());
        let mut pending = None;

        let observation = prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            EXPLICIT_BYTES,
            EXPLICIT_BYTES,
            0,
            &mut bytes,
        )
        .unwrap();

        assert!(observation.initialized());
        assert_eq!(observation.explicit_byte_len(), EXPLICIT_BYTES as u64);
        assert_eq!(observation.implicit_byte_offset(), EXPLICIT_BYTES as u64);
        assert_eq!(observation.implicit_byte_len(), 0);
        assert_eq!(bytes, original);
        assert_eq!(core.api.log, ["queue_create", "queue_async_error"]);

        launch_and_wait(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            &mut bytes,
        )
        .unwrap();
        assert_eq!(bytes, original);
        assert_eq!(
            core.api
                .log
                .iter()
                .filter(|operation| **operation == "queue_create")
                .count(),
            1
        );
        assert_eq!(
            core.api
                .log
                .iter()
                .filter(|operation| **operation == "publish")
                .count(),
            1
        );
    }

    #[test]
    fn implicit_initialization_rejects_layout_and_handle_substitution() {
        let (executable, mut kernel) = handles();
        for (explicit, offset, implicit) in [
            (48, 47, 256),
            (48, 49, 256),
            (48, 48, 0),
            (48, 48, 1),
            (48, 48, 255),
            (48, 48, 257),
            (32, 32, 256),
        ] {
            let mut core = make_core(MockApi::default());
            let mut pending = None;
            let mut bytes = kernarg();
            assert!(matches!(
                prepare_implicit_kernarg(
                    &mut core,
                    &mut pending,
                    &executable,
                    &kernel,
                    geometry(),
                    explicit,
                    offset,
                    implicit,
                    &mut bytes,
                ),
                Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(_))
            ));
            assert!(pending.is_none());
            assert!(core.api.log.is_empty());
        }
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let bounded_explicit = usize::try_from(MAX_ABI_BYTES).unwrap() + 1;
        let (bounded_executable, bounded_kernel) = handles_for_explicit_prefix(bounded_explicit);
        let mut bounded_bytes = kernarg_for_explicit_prefix(bounded_explicit);
        assert!(matches!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &bounded_executable,
                &bounded_kernel,
                geometry(),
                bounded_explicit,
                bounded_explicit,
                IMPLICIT_BYTES,
                &mut bounded_bytes,
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(_))
        ));
        assert!(pending.is_none());
        assert!(core.api.log.is_empty());

        assert!(matches!(
            validate_implicit_kernarg(
                &executable,
                &kernel,
                geometry(),
                usize::MAX,
                usize::MAX,
                IMPLICIT_BYTES,
                &[],
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
                "kernarg layout size overflow"
            ))
        ));

        kernel.executable_identity = HsaExecutableObjectIdentityV1::new([9; 32]).unwrap();
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        assert!(matches!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                48,
                48,
                256,
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(_))
        ));

        let (_, mut wrong_abi) = handles();
        wrong_abi.kernarg_segment_size = (TOTAL_BYTES - 1) as u32;
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        assert!(matches!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &wrong_abi,
                geometry(),
                48,
                48,
                256,
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(_))
        ));
        assert!(core.api.log.is_empty());
    }

    #[test]
    fn implicit_initialization_owns_one_exact_queue_binding() {
        let (executable, kernel) = handles();
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        assert!(matches!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                48,
                48,
                256,
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(_))
        ));
        assert_eq!(core.api.log, ["queue_create", "queue_async_error"]);
        destroy_pending_dispatch(&mut core.api, &mut pending);
        assert!(pending.is_none());
        assert!(core.api.log.ends_with(&["queue_destroy"]));
    }

    #[test]
    fn unload_cancels_prepublication_dispatch_before_executable_destruction() {
        let (executable, kernel) = handles();
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            EXPLICIT_BYTES,
            IMPLICIT_OFFSET,
            IMPLICIT_BYTES,
            &mut bytes,
        )
        .unwrap();
        drop(kernel);

        let observation = crate::lifecycle::unload_executable_after_pending_dispatch(
            &mut core,
            &mut pending,
            executable,
        )
        .unwrap();

        assert!(observation.released());
        assert!(pending.is_none());
        assert!(!core.api.log.contains(&"publish"));
        destroy_pending_dispatch(&mut core.api, &mut pending);
        assert_eq!(
            core.api.log,
            [
                "queue_create",
                "queue_async_error",
                "queue_destroy",
                "executable_destroy",
                "reader_destroy",
            ]
        );
        assert_eq!(
            core.api
                .log
                .iter()
                .filter(|operation| **operation == "queue_destroy")
                .count(),
            1
        );
    }

    #[test]
    fn prepared_queue_and_kernarg_cannot_cross_kernel_identities() {
        let (executable, first) = handles();
        let mut second = ReviewedHsaKernelV1 {
            symbol: 16,
            kernel_object: 17,
            executable_identity: first.executable_identity,
            identity: HsaKernelObjectIdentityV1::new([5; 32]).unwrap(),
            kernarg_segment_size: first.kernarg_segment_size,
            kernarg_segment_alignment: first.kernarg_segment_alignment,
            group_segment_size: first.group_segment_size,
            private_segment_size: first.private_segment_size,
        };
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &first,
            geometry(),
            EXPLICIT_BYTES,
            IMPLICIT_OFFSET,
            IMPLICIT_BYTES,
            &mut bytes,
        )
        .unwrap();

        assert!(matches!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &second,
                geometry(),
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::InvalidExecutableObservation(
                "dispatch handle, geometry, or kernarg binding"
            ))
        ));
        assert!(pending.is_none());
        assert_eq!(core.api.log.last(), Some(&"queue_destroy"));
        assert!(!core.api.log.contains(&"memory_allocate"));

        second.executable_identity = HsaExecutableObjectIdentityV1::new([9; 32]).unwrap();
        let mut fresh = kernarg();
        assert!(matches!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &second,
                geometry(),
                EXPLICIT_BYTES,
                IMPLICIT_OFFSET,
                IMPLICIT_BYTES,
                &mut fresh,
            ),
            Err(HsaRuntimeAdapterError::InvalidImplicitKernarg(
                "kernel/executable identity"
            ))
        ));
        assert!(pending.is_none());
    }

    #[test]
    fn implicit_queue_creation_and_observation_fail_closed() {
        let (executable, kernel) = handles();
        let mut api = MockApi::default();
        api.failures.insert("queue_create", 71);
        let mut core = make_core(api);
        let mut pending = None;
        let mut bytes = kernarg();
        assert!(
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                48,
                48,
                256,
                &mut bytes,
            )
            .is_err()
        );
        assert_eq!(core.api.log, ["queue_create"]);
    }

    #[test]
    fn synchronous_dispatch_retains_resources_until_zero_completion() {
        let (executable, kernel) = handles();
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        let observation = launch_and_wait(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            &mut bytes,
        )
        .unwrap();
        assert!(observation.completed());
        assert!(pending.is_none());
        assert_eq!(core.api.published_grid, Some([512, 1, 1]));
        assert_eq!(core.api.written, bytes);
        assert_eq!(
            core.api.log,
            [
                "queue_create",
                "queue_async_error",
                "memory_allocate",
                "allow_access",
                "write_memory",
                "signal_create",
                "queue_async_error",
                "publish",
                "signal_load",
                "queue_async_error",
                "signal_destroy",
                "memory_free",
                "queue_destroy",
            ]
        );
    }

    #[test]
    fn prepublication_failures_clean_every_live_resource_in_reverse_order() {
        let cases = [
            (
                "memory_allocate",
                vec![
                    "queue_create",
                    "queue_async_error",
                    "memory_allocate",
                    "queue_destroy",
                ],
            ),
            (
                "allow_access",
                vec![
                    "queue_create",
                    "queue_async_error",
                    "memory_allocate",
                    "allow_access",
                    "memory_free",
                    "queue_destroy",
                ],
            ),
            (
                "signal_create",
                vec![
                    "queue_create",
                    "queue_async_error",
                    "memory_allocate",
                    "allow_access",
                    "write_memory",
                    "signal_create",
                    "memory_free",
                    "queue_destroy",
                ],
            ),
        ];
        for (failure, expected) in cases {
            let (executable, kernel) = handles();
            let mut api = MockApi::default();
            api.failures.insert(failure, 77);
            let mut core = make_core(api);
            let mut pending = None;
            let mut bytes = kernarg();
            prepare_implicit_kernarg(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                48,
                48,
                256,
                &mut bytes,
            )
            .unwrap();
            assert!(
                launch_and_wait(
                    &mut core,
                    &mut pending,
                    &executable,
                    &kernel,
                    geometry(),
                    &mut bytes,
                )
                .is_err()
            );
            assert_eq!(core.api.log, expected, "failure edge {failure}");
        }

        let (executable, kernel) = handles();
        let api = MockApi {
            fail_async_call: Some(2),
            ..MockApi::default()
        };
        let mut core = make_core(api);
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        assert!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                &mut bytes,
            )
            .is_err()
        );
        assert!(core.api.log.ends_with(&[
            "signal_create",
            "queue_async_error",
            "signal_destroy",
            "memory_free",
            "queue_destroy",
        ]));
    }

    #[test]
    fn dispatch_allocation_must_meet_the_resolved_kernel_alignment() {
        let (executable, mut kernel) = handles();
        kernel.kernarg_segment_alignment = 8192;
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        assert!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                &mut bytes,
            )
            .is_err()
        );
        assert!(!core.api.log.contains(&"publish"));
        assert!(core.api.log.ends_with(&["memory_free", "queue_destroy"]));
    }

    #[test]
    fn launch_rejects_kernarg_or_geometry_substitution_before_publication() {
        let (executable, kernel) = handles();
        let mut core = make_core(MockApi::default());
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        bytes[0] ^= 1;
        assert!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                &mut bytes,
            )
            .is_err()
        );
        assert!(pending.is_none());
        assert_eq!(core.api.log.last(), Some(&"queue_destroy"));
        assert!(!core.api.log.contains(&"memory_allocate"));
    }

    #[test]
    fn publication_failure_is_definitely_presubmit_and_cleans_resources() {
        let (executable, kernel) = handles();
        let mut api = MockApi::default();
        api.failures.insert("publish", 79);
        let mut core = make_core(api);
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        assert!(matches!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::RuntimeCall { status: 79, .. })
        ));
        assert!(
            core.api
                .log
                .ends_with(&["signal_destroy", "memory_free", "queue_destroy"])
        );
    }

    fn submitted_dispatch() -> SubmittedDispatch {
        SubmittedDispatch {
            resources: PreSubmitDispatch {
                queue: QueueHandle::for_test(0xabc0, 0, 64),
                kernarg_address: 0x1000,
                completion_signal: 51,
            },
            packet_id: 61,
        }
    }

    #[test]
    fn spurious_wakeups_repeat_until_exact_zero_completion() {
        let mut api = MockApi {
            completion_sequence: VecDeque::from([1, -1, 0]),
            ..MockApi::default()
        };
        assert!(matches!(
            await_quiescence(&mut api, submitted_dispatch(), COMPLETION_TIMEOUT),
            CompletionTransition::Quiesced {
                queue_error: None,
                ..
            }
        ));
        assert_eq!(
            api.log,
            [
                "signal_load",
                "queue_async_error",
                "signal_load",
                "queue_async_error",
                "signal_load",
                "queue_async_error",
            ]
        );
    }

    #[test]
    fn queue_fault_and_completion_deadline_remain_submitted_without_cleanup() {
        let mut faulted = MockApi {
            completion: 1,
            fail_async_call: Some(1),
            ..MockApi::default()
        };
        match await_quiescence(&mut faulted, submitted_dispatch(), COMPLETION_TIMEOUT) {
            CompletionTransition::Unquiesced(unquiesced) => assert!(matches!(
                unquiesced.reason,
                UnquiescedReason::QueueError(ApiError { status: 82, .. })
            )),
            CompletionTransition::Quiesced { .. } => panic!("faulted queue reported quiescence"),
        }
        assert!(!faulted.log.contains(&"signal_destroy"));
        assert!(!faulted.log.contains(&"memory_free"));
        assert!(!faulted.log.contains(&"queue_destroy"));

        let mut timed_out = MockApi {
            completion: 1,
            ..MockApi::default()
        };
        match await_quiescence(&mut timed_out, submitted_dispatch(), Duration::ZERO) {
            CompletionTransition::Unquiesced(unquiesced) => assert_eq!(
                unquiesced.reason,
                UnquiescedReason::CompletionDeadline {
                    last_observation: 1
                }
            ),
            CompletionTransition::Quiesced { .. } => {
                panic!("expired completion deadline reported quiescence")
            }
        }
        assert_eq!(timed_out.log, ["signal_load", "queue_async_error"]);
        assert!(!timed_out.log.contains(&"memory_free"));
    }

    #[test]
    fn completed_async_error_is_reported_after_conclusive_cleanup() {
        let (executable, kernel) = handles();
        let api = MockApi {
            fail_async_call: Some(3),
            ..MockApi::default()
        };
        let mut core = make_core(api);
        let mut pending = None;
        let mut bytes = kernarg();
        prepare_implicit_kernarg(
            &mut core,
            &mut pending,
            &executable,
            &kernel,
            geometry(),
            48,
            48,
            256,
            &mut bytes,
        )
        .unwrap();
        assert!(matches!(
            launch_and_wait(
                &mut core,
                &mut pending,
                &executable,
                &kernel,
                geometry(),
                &mut bytes,
            ),
            Err(HsaRuntimeAdapterError::RuntimeCall { status: 82, .. })
        ));
        assert!(
            core.api
                .log
                .ends_with(&["signal_destroy", "memory_free", "queue_destroy"])
        );
    }

    #[test]
    #[cfg(unix)]
    fn ambiguous_dispatch_cleanup_is_terminal() {
        const CHILD: &str = "FE2O3_HSA_AMBIGUOUS_DISPATCH_CLEANUP_CHILD";
        if let Ok(case) = std::env::var(CHILD) {
            let (executable, kernel) = handles();
            let mut api = MockApi::default();
            match case.as_str() {
                "implicit-queue" => {
                    api.fail_async_call = Some(1);
                    api.failures.insert("queue_destroy", 73);
                    let mut core = make_core(api);
                    let mut pending = None;
                    let mut bytes = kernarg();
                    let _ = prepare_implicit_kernarg(
                        &mut core,
                        &mut pending,
                        &executable,
                        &kernel,
                        geometry(),
                        48,
                        48,
                        256,
                        &mut bytes,
                    );
                }
                "presubmit-signal" | "quiesced-signal" => {
                    if case == "quiesced-signal" {
                        api.fail_async_call = Some(3);
                    } else {
                        api.failures.insert("publish", 74);
                    }
                    api.failures.insert("signal_destroy", 75);
                    let mut core = make_core(api);
                    let mut pending = None;
                    let mut bytes = kernarg();
                    prepare_implicit_kernarg(
                        &mut core,
                        &mut pending,
                        &executable,
                        &kernel,
                        geometry(),
                        48,
                        48,
                        256,
                        &mut bytes,
                    )
                    .unwrap();
                    let _ = launch_and_wait(
                        &mut core,
                        &mut pending,
                        &executable,
                        &kernel,
                        geometry(),
                        &mut bytes,
                    );
                }
                "pending-unload-queue" | "pending-unload-executable" | "pending-unload-reader" => {
                    let mut core = make_core(api);
                    let mut pending = None;
                    let mut bytes = kernarg();
                    prepare_implicit_kernarg(
                        &mut core,
                        &mut pending,
                        &executable,
                        &kernel,
                        geometry(),
                        EXPLICIT_BYTES,
                        IMPLICIT_OFFSET,
                        IMPLICIT_BYTES,
                        &mut bytes,
                    )
                    .unwrap();
                    let failure = match case.as_str() {
                        "pending-unload-queue" => "queue_destroy",
                        "pending-unload-executable" => "executable_destroy",
                        "pending-unload-reader" => "reader_destroy",
                        _ => unreachable!(),
                    };
                    core.api.failures.insert(failure, 76);
                    drop(kernel);
                    let _ = crate::lifecycle::unload_executable_after_pending_dispatch(
                        &mut core,
                        &mut pending,
                        executable,
                    );
                }
                _ => panic!("unknown dispatch cleanup case"),
            }
            std::process::exit(91);
        }

        use std::os::unix::process::ExitStatusExt;
        for case in [
            "implicit-queue",
            "presubmit-signal",
            "quiesced-signal",
            "pending-unload-queue",
            "pending-unload-executable",
            "pending-unload-reader",
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .arg("--exact")
                .arg("dispatch::tests::ambiguous_dispatch_cleanup_is_terminal")
                .arg("--nocapture")
                .env(CHILD, case);
            let status = crate::test_process_execution::status(&mut command).unwrap();
            assert_eq!(status.signal(), Some(6), "cleanup case {case}: {status}");
        }
    }

    #[cfg(feature = "hardware-test-hooks")]
    #[test]
    fn profiled_dispatch_duration_uses_wide_integer_arithmetic() {
        let start = (1_u64 << 63) + 7;
        let observation = ReviewedHsaProfiledDispatchObservationV1 {
            start_tick: start,
            end_tick: start + 123_456_789,
            timestamp_frequency_hz: 100_000_000,
            packet_id: u64::MAX,
        };
        assert_eq!(observation.start_tick(), start);
        assert_eq!(observation.end_tick(), start + 123_456_789);
        assert_eq!(observation.timestamp_frequency_hz(), 100_000_000);
        assert_eq!(observation.packet_id(), u64::MAX);
        assert_eq!(observation.duration_ns(), 1_234_567_890);
    }

    #[allow(dead_code)]
    fn _payload_type_is_part_of_the_reviewed_surface(_: PayloadDigest) {}
}
