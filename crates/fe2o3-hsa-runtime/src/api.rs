use crate::sys;
use fe2o3_artifacts::{DigestAlgorithm, PayloadDigest};
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString};
use std::fs;
use std::io::Read;
use std::mem::MaybeUninit;
#[cfg(fe2o3_hsa_runtime)]
use std::os::unix::fs::MetadataExt;

pub(crate) const HSA_SUCCESS: i32 = 0;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeFacts {
    pub version_major: u16,
    pub version_minor: u16,
    pub image_digest: PayloadDigest,
    pub instance: [u8; 16],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HipFacts {
    pub uuid: [u8; 16],
    pub pci_bus_id: String,
    pub round_trip_ordinal: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AgentFacts {
    pub handle: u64,
    pub node: u32,
    pub device_type: u32,
    pub feature: u32,
    pub profile: u32,
    pub queue_min_size: u32,
    pub queue_max_size: u32,
    pub queue_type: u32,
    pub domain: u32,
    pub bdf_id: u32,
    pub name: String,
    pub uuid: String,
    pub isa: String,
    pub matching_isa_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PoolFacts {
    pub handle: u64,
    pub owner_agent: u64,
    pub owner_node: u32,
    pub segment: u32,
    pub global_flags: u32,
    pub runtime_alloc_allowed: bool,
    pub runtime_alloc_alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SymbolFacts {
    pub handle: u64,
    pub kernel_object: u64,
    pub kind: u32,
    pub kernarg_size: u32,
    pub kernarg_alignment: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub name: String,
}

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DispatchTimeFacts {
    pub start: u64,
    pub end: u64,
}

pub(crate) struct QueueHandle {
    raw: sys::QueueRecord,
}

impl QueueHandle {
    pub const fn id(&self) -> u64 {
        self.raw.id
    }

    pub const fn pointer(&self) -> usize {
        self.raw.pointer
    }

    #[cfg(test)]
    pub const fn for_test(pointer: usize, id: u64, size: u32) -> Self {
        Self {
            raw: sys::QueueRecord {
                pointer,
                id,
                size,
                async_error: 1,
            },
        }
    }
}

const fn valid_queue_record(record: &sys::QueueRecord) -> bool {
    record.pointer != 0
        && record.size != 0
        && record.size.is_power_of_two()
        && record.async_error != 0
}

fn queue_record_after_create(
    status: i32,
    record: sys::QueueRecord,
) -> Result<sys::QueueRecord, ApiError> {
    if status == HSA_SUCCESS {
        return Ok(record);
    }
    if record.pointer != 0 || record.async_error != 0 {
        std::process::abort();
    }
    Err(ApiError {
        operation: "hsa_queue_create",
        status,
    })
}

pub(crate) trait EnvironmentApi {
    fn initialize(&mut self) -> Result<RuntimeFacts, ApiError>;
    fn shut_down(&mut self) -> Result<(), ApiError>;
    fn observe_hip_device(&mut self, ordinal: i32) -> Result<HipFacts, ApiError>;
    fn collect_agents(&mut self) -> Result<Vec<AgentFacts>, ApiError>;
    fn collect_kernarg_pools(&mut self) -> Result<Vec<PoolFacts>, ApiError>;
}

pub(crate) trait ExecutableApi: EnvironmentApi {
    fn reader_create(&mut self, bytes: &[u8]) -> Result<u64, ApiError>;
    fn reader_destroy(&mut self, reader: u64) -> Result<(), ApiError>;
    fn executable_create(&mut self, profile: u32) -> Result<u64, ApiError>;
    fn executable_load(
        &mut self,
        executable: u64,
        agent: u64,
        reader: u64,
    ) -> Result<u64, ApiError>;
    fn executable_freeze(&mut self, executable: u64) -> Result<(), ApiError>;
    fn executable_destroy(&mut self, executable: u64) -> Result<(), ApiError>;
    fn resolve_symbol(
        &mut self,
        executable: u64,
        agent: u64,
        name: &str,
    ) -> Result<SymbolFacts, ApiError>;
}

pub(crate) trait DispatchApi: ExecutableApi {
    fn memory_allocate(&mut self, pool: u64, len: usize) -> Result<usize, ApiError>;
    fn allow_access(&mut self, agent: u64, address: usize) -> Result<(), ApiError>;
    fn write_memory(&mut self, address: usize, bytes: &[u8]);
    fn memory_free(&mut self, address: usize) -> Result<(), ApiError>;
    fn queue_create(&mut self, agent: u64, size: u32) -> Result<QueueHandle, ApiError>;
    fn queue_async_error(&mut self, queue: &QueueHandle) -> Result<(), ApiError>;
    fn queue_destroy(&mut self, queue: &mut QueueHandle) -> Result<(), ApiError>;
    fn signal_create(&mut self, initial_value: i64) -> Result<u64, ApiError>;
    fn signal_destroy(&mut self, signal: u64) -> Result<(), ApiError>;
    fn signal_load_acquire(&mut self, signal: u64) -> i64;
    #[cfg(feature = "hardware-test-hooks")]
    fn queue_enable_profiling(&mut self, _queue: &QueueHandle) -> Result<(), ApiError> {
        Err(ApiError {
            operation: "HSA queue profiling unavailable",
            status: -1,
        })
    }
    #[cfg(feature = "hardware-test-hooks")]
    fn signal_store_release(&mut self, _signal: u64, _value: i64) -> Result<(), ApiError> {
        Err(ApiError {
            operation: "HSA signal store unavailable",
            status: -1,
        })
    }
    #[cfg(feature = "hardware-test-hooks")]
    fn timestamp_frequency(&mut self) -> Result<u64, ApiError> {
        Err(ApiError {
            operation: "HSA timestamp frequency unavailable",
            status: -1,
        })
    }
    #[cfg(feature = "hardware-test-hooks")]
    fn dispatch_time(&mut self, _agent: u64, _signal: u64) -> Result<DispatchTimeFacts, ApiError> {
        Err(ApiError {
            operation: "HSA dispatch profiling unavailable",
            status: -1,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn publish_dispatch(
        &mut self,
        queue: &QueueHandle,
        grid: [u32; 3],
        workgroup: [u32; 3],
        private_segment_size: u32,
        group_segment_size: u32,
        kernel_object: u64,
        kernarg: usize,
        completion_signal: u64,
    ) -> Result<u64, ApiError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ApiError {
    pub operation: &'static str,
    pub status: i32,
}

impl ApiError {
    fn status(operation: &'static str, status: i32) -> Result<(), Self> {
        if status == HSA_SUCCESS {
            Ok(())
        } else {
            Err(Self { operation, status })
        }
    }
}

pub(crate) struct DirectRuntimeApi {
    initialized: bool,
}

impl DirectRuntimeApi {
    pub const fn new() -> Self {
        Self { initialized: false }
    }
}

#[cfg(fe2o3_hsa_runtime)]
impl EnvironmentApi for DirectRuntimeApi {
    fn initialize(&mut self) -> Result<RuntimeFacts, ApiError> {
        if self.initialized {
            return Err(ApiError {
                operation: "hsa_init duplicate adapter initialization",
                status: -1,
            });
        }
        // SAFETY: HSA initialization has no caller-owned pointer arguments.
        ApiError::status("hsa_init", unsafe { sys::fe2o3_hsa_init() })?;
        self.initialized = true;

        let result = (|| {
            let mut major = 0;
            let mut minor = 0;
            // SAFETY: both outputs are live writable integers.
            ApiError::status("hsa_system_get_info(version)", unsafe {
                sys::fe2o3_hsa_runtime_version(&mut major, &mut minor)
            })?;
            // SAFETY: the shim returns addresses from the linked HSA and HIP images.
            let hsa_address = unsafe { sys::fe2o3_hsa_runtime_function_address() };
            // SAFETY: the shim returns the address of the linked hipInit function.
            let hip_address = unsafe { sys::fe2o3_hip_runtime_function_address() };
            let hsa_digest = measure_loaded_runtime_image(hsa_address)?;
            let hip_digest = measure_loaded_runtime_image(hip_address)?;
            let image_digest = derive_runtime_stack_digest(hsa_digest, hip_digest);
            let instance = derive_runtime_instance(image_digest, hsa_address, hip_address);
            Ok(RuntimeFacts {
                version_major: major,
                version_minor: minor,
                image_digest,
                instance,
            })
        })();
        if result.is_err() {
            let cleanup = self.shut_down();
            if cleanup.is_err() {
                std::process::abort();
            }
        }
        result
    }

    fn shut_down(&mut self) -> Result<(), ApiError> {
        if !self.initialized {
            return Ok(());
        }
        // SAFETY: this object owns exactly one successful hsa_init reference.
        ApiError::status("hsa_shut_down", unsafe { sys::fe2o3_hsa_shut_down() })?;
        self.initialized = false;
        Ok(())
    }

    fn observe_hip_device(&mut self, ordinal: i32) -> Result<HipFacts, ApiError> {
        let mut record = MaybeUninit::<sys::HipDeviceRecord>::zeroed();
        // SAFETY: the output points to enough writable storage for the shim record.
        ApiError::status("HIP device UUID/PCI observation", unsafe {
            sys::fe2o3_hip_observe_device(ordinal, record.as_mut_ptr())
        })?;
        // SAFETY: successful completion initializes the complete output record.
        let record = unsafe { record.assume_init() };
        Ok(HipFacts {
            uuid: record.uuid,
            pci_bus_id: c_text(&record.pci_bus_id, "HIP PCI bus ID")?,
            round_trip_ordinal: record.round_trip_ordinal,
        })
    }

    fn collect_agents(&mut self) -> Result<Vec<AgentFacts>, ApiError> {
        let mut records =
            [const { MaybeUninit::<sys::AgentRecord>::zeroed() }; sys::AGENT_CAPACITY];
        let mut count = 0;
        // SAFETY: the output array has exactly the capacity reported to the shim.
        ApiError::status("hsa_iterate_agents", unsafe {
            sys::fe2o3_hsa_collect_agents(
                records.as_mut_ptr().cast(),
                sys::AGENT_CAPACITY as u32,
                &mut count,
            )
        })?;
        let count = usize::try_from(count).map_err(|_| ApiError {
            operation: "HSA agent count conversion",
            status: -1,
        })?;
        if count > records.len() {
            return Err(ApiError {
                operation: "HSA agent count exceeds reviewed capacity",
                status: -1,
            });
        }
        records[..count]
            .iter()
            .map(|record| {
                // SAFETY: the shim initializes every record below the returned count.
                let record = unsafe { record.assume_init_ref() };
                Ok(AgentFacts {
                    handle: record.handle,
                    node: record.node,
                    device_type: record.device_type,
                    feature: record.feature,
                    profile: record.profile,
                    queue_min_size: record.queue_min_size,
                    queue_max_size: record.queue_max_size,
                    queue_type: record.queue_type,
                    domain: record.domain,
                    bdf_id: record.bdf_id,
                    name: c_text(&record.name, "HSA agent name")?,
                    uuid: c_text(&record.uuid, "HSA agent UUID")?,
                    isa: c_text(&record.isa, "HSA ISA name")?,
                    matching_isa_count: record.matching_isa_count,
                })
            })
            .collect()
    }

    fn collect_kernarg_pools(&mut self) -> Result<Vec<PoolFacts>, ApiError> {
        let mut records = [const { MaybeUninit::<sys::PoolRecord>::zeroed() }; sys::POOL_CAPACITY];
        let mut count = 0;
        // SAFETY: the output array has exactly the capacity reported to the shim.
        ApiError::status("hsa_amd_agent_iterate_memory_pools", unsafe {
            sys::fe2o3_hsa_collect_kernarg_pools(
                records.as_mut_ptr().cast(),
                sys::POOL_CAPACITY as u32,
                &mut count,
            )
        })?;
        let count = usize::try_from(count).map_err(|_| ApiError {
            operation: "HSA memory-pool count conversion",
            status: -1,
        })?;
        if count > records.len() {
            return Err(ApiError {
                operation: "HSA memory-pool count exceeds reviewed capacity",
                status: -1,
            });
        }
        Ok(records[..count]
            .iter()
            .map(|record| {
                // SAFETY: the shim initializes every record below the returned count.
                let record = unsafe { record.assume_init_ref() };
                PoolFacts {
                    handle: record.handle,
                    owner_agent: record.owner_agent,
                    owner_node: record.owner_node,
                    segment: record.segment,
                    global_flags: record.global_flags,
                    runtime_alloc_allowed: record.runtime_alloc_allowed == 1,
                    runtime_alloc_alignment: record.runtime_alloc_alignment,
                }
            })
            .collect())
    }
}

#[cfg(fe2o3_hsa_runtime)]
impl ExecutableApi for DirectRuntimeApi {
    fn reader_create(&mut self, bytes: &[u8]) -> Result<u64, ApiError> {
        if bytes.is_empty() {
            return Err(ApiError {
                operation: "reject empty HSA code object",
                status: -1,
            });
        }
        let mut reader = 0;
        // SAFETY: `bytes` remains live in the executable token until reader destruction.
        ApiError::status("hsa_code_object_reader_create_from_memory", unsafe {
            sys::fe2o3_hsa_reader_create(bytes.as_ptr().cast(), bytes.len(), &mut reader)
        })?;
        if reader == 0 {
            return Err(ApiError {
                operation: "validate HSA code object reader handle",
                status: -1,
            });
        }
        Ok(reader)
    }

    fn reader_destroy(&mut self, reader: u64) -> Result<(), ApiError> {
        // SAFETY: the lifecycle passes one live reader handle exactly once.
        ApiError::status("hsa_code_object_reader_destroy", unsafe {
            sys::fe2o3_hsa_reader_destroy(reader)
        })
    }

    fn executable_create(&mut self, profile: u32) -> Result<u64, ApiError> {
        let mut executable = 0;
        // SAFETY: the output is writable and `profile` came from the selected agent.
        ApiError::status("hsa_executable_create_alt", unsafe {
            sys::fe2o3_hsa_executable_create(profile, &mut executable)
        })?;
        if executable == 0 {
            return Err(ApiError {
                operation: "validate HSA executable handle",
                status: -1,
            });
        }
        Ok(executable)
    }

    fn executable_load(
        &mut self,
        executable: u64,
        agent: u64,
        reader: u64,
    ) -> Result<u64, ApiError> {
        let mut loaded = 0;
        // SAFETY: all handles are private live values owned by this lifecycle.
        ApiError::status("hsa_executable_load_agent_code_object", unsafe {
            sys::fe2o3_hsa_executable_load(executable, agent, reader, &mut loaded)
        })?;
        if loaded == 0 {
            return Err(ApiError {
                operation: "validate HSA loaded code object handle",
                status: -1,
            });
        }
        Ok(loaded)
    }

    fn executable_freeze(&mut self, executable: u64) -> Result<(), ApiError> {
        // SAFETY: the lifecycle passes one live unfrozen executable.
        ApiError::status("hsa_executable_freeze", unsafe {
            sys::fe2o3_hsa_executable_freeze(executable)
        })
    }

    fn executable_destroy(&mut self, executable: u64) -> Result<(), ApiError> {
        // SAFETY: the lifecycle has quiesced dispatch and passes the handle once.
        ApiError::status("hsa_executable_destroy", unsafe {
            sys::fe2o3_hsa_executable_destroy(executable)
        })
    }

    fn resolve_symbol(
        &mut self,
        executable: u64,
        agent: u64,
        name: &str,
    ) -> Result<SymbolFacts, ApiError> {
        let name = CString::new(name).map_err(|_| ApiError {
            operation: "reject HSA symbol containing NUL",
            status: -1,
        })?;
        let mut record = MaybeUninit::<sys::SymbolRecord>::zeroed();
        // SAFETY: handles are live, the name is terminated, and output storage is writable.
        ApiError::status("hsa_executable_get_symbol_by_name", unsafe {
            sys::fe2o3_hsa_resolve_symbol(executable, agent, name.as_ptr(), record.as_mut_ptr())
        })?;
        // SAFETY: successful completion initializes the complete output record.
        let record = unsafe { record.assume_init() };
        Ok(SymbolFacts {
            handle: record.handle,
            kernel_object: record.kernel_object,
            kind: record.kind,
            kernarg_size: record.kernarg_size,
            kernarg_alignment: record.kernarg_alignment,
            group_segment_size: record.group_segment_size,
            private_segment_size: record.private_segment_size,
            name: c_text(&record.name, "HSA executable symbol name")?,
        })
    }
}

#[cfg(fe2o3_hsa_runtime)]
impl DispatchApi for DirectRuntimeApi {
    fn memory_allocate(&mut self, pool: u64, len: usize) -> Result<usize, ApiError> {
        let mut address = core::ptr::null_mut();
        // SAFETY: the private pool handle was selected from live HSA observations.
        ApiError::status("hsa_amd_memory_pool_allocate(kernarg)", unsafe {
            sys::fe2o3_hsa_pool_allocate(pool, len, &mut address)
        })?;
        if address.is_null() {
            return Err(ApiError {
                operation: "validate HSA kernarg allocation",
                status: -1,
            });
        }
        Ok(address.addr())
    }

    fn allow_access(&mut self, agent: u64, address: usize) -> Result<(), ApiError> {
        // SAFETY: the allocation and agent are private live handles from this runtime.
        ApiError::status("hsa_amd_agents_allow_access(kernarg)", unsafe {
            sys::fe2o3_hsa_allow_access(agent, address as *mut core::ffi::c_void)
        })
    }

    fn write_memory(&mut self, address: usize, bytes: &[u8]) {
        // SAFETY: `address` denotes a live allocation of exactly `bytes.len()`
        // bytes retained by the dispatch lifecycle, and the regions do not overlap.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
    }

    fn memory_free(&mut self, address: usize) -> Result<(), ApiError> {
        // SAFETY: the lifecycle passes a quiesced live allocation exactly once.
        ApiError::status("hsa_amd_memory_pool_free(kernarg)", unsafe {
            sys::fe2o3_hsa_memory_free(address as *mut core::ffi::c_void)
        })
    }

    fn queue_create(&mut self, agent: u64, size: u32) -> Result<QueueHandle, ApiError> {
        let mut raw = MaybeUninit::<sys::QueueRecord>::zeroed();
        // SAFETY: the output is writable and the agent is a selected kernel agent.
        let status = unsafe { sys::fe2o3_hsa_queue_create(agent, size, raw.as_mut_ptr()) };
        // SAFETY: the shim zeroes the complete record before any HSA call and
        // preserves live native authority in it when cleanup is ambiguous.
        let mut raw = queue_record_after_create(status, unsafe { raw.assume_init() })?;
        if !valid_queue_record(&raw) {
            if raw.pointer != 0 && raw.async_error != 0 {
                // SAFETY: successful creation returned this live private record.
                let cleanup = unsafe { sys::fe2o3_hsa_queue_destroy(&mut raw) };
                if cleanup != HSA_SUCCESS {
                    std::process::abort();
                }
            }
            return Err(ApiError {
                operation: "validate HSA queue observation",
                status: -1,
            });
        }
        Ok(QueueHandle { raw })
    }

    fn queue_async_error(&mut self, queue: &QueueHandle) -> Result<(), ApiError> {
        // SAFETY: the queue record owns a live callback-state allocation.
        ApiError::status("HSA queue asynchronous status", unsafe {
            sys::fe2o3_hsa_queue_async_error(&queue.raw)
        })
    }

    fn queue_destroy(&mut self, queue: &mut QueueHandle) -> Result<(), ApiError> {
        // SAFETY: dispatch has completed or was never submitted; this record is destroyed once.
        ApiError::status("hsa_queue_destroy", unsafe {
            sys::fe2o3_hsa_queue_destroy(&mut queue.raw)
        })
    }

    fn signal_create(&mut self, initial_value: i64) -> Result<u64, ApiError> {
        let mut signal = 0;
        // SAFETY: the output points to one writable signal handle.
        ApiError::status("hsa_signal_create", unsafe {
            sys::fe2o3_hsa_signal_create(initial_value, &mut signal)
        })?;
        if signal == 0 {
            return Err(ApiError {
                operation: "validate HSA completion signal",
                status: -1,
            });
        }
        Ok(signal)
    }

    fn signal_destroy(&mut self, signal: u64) -> Result<(), ApiError> {
        // SAFETY: dispatch completion is observed and the private signal is destroyed once.
        ApiError::status("hsa_signal_destroy", unsafe {
            sys::fe2o3_hsa_signal_destroy(signal)
        })
    }

    fn signal_load_acquire(&mut self, signal: u64) -> i64 {
        // SAFETY: the private completion signal remains live throughout this
        // nonblocking acquire load.
        unsafe { sys::fe2o3_hsa_signal_load_acquire(signal) }
    }

    #[cfg(feature = "hardware-test-hooks")]
    fn queue_enable_profiling(&mut self, queue: &QueueHandle) -> Result<(), ApiError> {
        ApiError::status("hsa_amd_profiling_set_profiler_enabled", unsafe {
            sys::fe2o3_hsa_queue_enable_profiling(&queue.raw)
        })
    }

    #[cfg(feature = "hardware-test-hooks")]
    fn signal_store_release(&mut self, signal: u64, value: i64) -> Result<(), ApiError> {
        ApiError::status("hsa_signal_store_screlease", unsafe {
            sys::fe2o3_hsa_signal_store_release(signal, value)
        })
    }

    #[cfg(feature = "hardware-test-hooks")]
    fn timestamp_frequency(&mut self) -> Result<u64, ApiError> {
        let mut frequency = 0;
        ApiError::status("hsa_system_get_info(timestamp frequency)", unsafe {
            sys::fe2o3_hsa_system_timestamp_frequency(&mut frequency)
        })?;
        if !(1..=10_000_000_000).contains(&frequency) {
            return Err(ApiError {
                operation: "validate HSA timestamp frequency",
                status: -1,
            });
        }
        Ok(frequency)
    }

    #[cfg(feature = "hardware-test-hooks")]
    fn dispatch_time(&mut self, agent: u64, signal: u64) -> Result<DispatchTimeFacts, ApiError> {
        let mut record = MaybeUninit::<sys::DispatchTimeRecord>::zeroed();
        ApiError::status("hsa_amd_profiling_get_dispatch_time", unsafe {
            sys::fe2o3_hsa_dispatch_time(agent, signal, record.as_mut_ptr())
        })?;
        let record = unsafe { record.assume_init() };
        if record.end < record.start {
            return Err(ApiError {
                operation: "validate HSA dispatch timestamps",
                status: -1,
            });
        }
        Ok(DispatchTimeFacts {
            start: record.start,
            end: record.end,
        })
    }

    fn publish_dispatch(
        &mut self,
        queue: &QueueHandle,
        grid: [u32; 3],
        workgroup: [u32; 3],
        private_segment_size: u32,
        group_segment_size: u32,
        kernel_object: u64,
        kernarg: usize,
        completion_signal: u64,
    ) -> Result<u64, ApiError> {
        let mut packet_id = 0;
        // SAFETY: every handle and pointer is private, live, and retained until completion.
        ApiError::status("publish HSA AQL kernel dispatch", unsafe {
            sys::fe2o3_hsa_publish_kernel_dispatch(
                &queue.raw,
                grid.as_ptr(),
                workgroup.as_ptr(),
                private_segment_size,
                group_segment_size,
                kernel_object,
                kernarg as *mut core::ffi::c_void,
                completion_signal,
                &mut packet_id,
            )
        })?;
        Ok(packet_id)
    }
}

#[cfg(not(fe2o3_hsa_runtime))]
impl EnvironmentApi for DirectRuntimeApi {
    fn initialize(&mut self) -> Result<RuntimeFacts, ApiError> {
        Err(ApiError {
            operation: "ROCm HSA runtime availability",
            status: -1,
        })
    }

    fn shut_down(&mut self) -> Result<(), ApiError> {
        Ok(())
    }

    fn observe_hip_device(&mut self, _ordinal: i32) -> Result<HipFacts, ApiError> {
        unreachable!("initialization fails before HIP observation")
    }

    fn collect_agents(&mut self) -> Result<Vec<AgentFacts>, ApiError> {
        unreachable!("initialization fails before HSA agent enumeration")
    }

    fn collect_kernarg_pools(&mut self) -> Result<Vec<PoolFacts>, ApiError> {
        unreachable!("initialization fails before HSA pool enumeration")
    }
}

#[cfg(not(fe2o3_hsa_runtime))]
impl ExecutableApi for DirectRuntimeApi {
    fn reader_create(&mut self, _bytes: &[u8]) -> Result<u64, ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn reader_destroy(&mut self, _reader: u64) -> Result<(), ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn executable_create(&mut self, _profile: u32) -> Result<u64, ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn executable_load(
        &mut self,
        _executable: u64,
        _agent: u64,
        _reader: u64,
    ) -> Result<u64, ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn executable_freeze(&mut self, _executable: u64) -> Result<(), ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn executable_destroy(&mut self, _executable: u64) -> Result<(), ApiError> {
        unreachable!("initialization fails before executable operations")
    }

    fn resolve_symbol(
        &mut self,
        _executable: u64,
        _agent: u64,
        _name: &str,
    ) -> Result<SymbolFacts, ApiError> {
        unreachable!("initialization fails before executable operations")
    }
}

#[cfg(not(fe2o3_hsa_runtime))]
impl DispatchApi for DirectRuntimeApi {
    fn memory_allocate(&mut self, _pool: u64, _len: usize) -> Result<usize, ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn allow_access(&mut self, _agent: u64, _address: usize) -> Result<(), ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn write_memory(&mut self, _address: usize, _bytes: &[u8]) {
        unreachable!("initialization fails before dispatch operations")
    }

    fn memory_free(&mut self, _address: usize) -> Result<(), ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn queue_create(&mut self, _agent: u64, _size: u32) -> Result<QueueHandle, ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn queue_async_error(&mut self, _queue: &QueueHandle) -> Result<(), ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn queue_destroy(&mut self, _queue: &mut QueueHandle) -> Result<(), ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn signal_create(&mut self, _initial_value: i64) -> Result<u64, ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn signal_destroy(&mut self, _signal: u64) -> Result<(), ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }

    fn signal_load_acquire(&mut self, _signal: u64) -> i64 {
        unreachable!("initialization fails before dispatch operations")
    }

    fn publish_dispatch(
        &mut self,
        _queue: &QueueHandle,
        _grid: [u32; 3],
        _workgroup: [u32; 3],
        _private_segment_size: u32,
        _group_segment_size: u32,
        _kernel_object: u64,
        _kernarg: usize,
        _completion_signal: u64,
    ) -> Result<u64, ApiError> {
        unreachable!("initialization fails before dispatch operations")
    }
}

#[cfg(fe2o3_hsa_runtime)]
fn c_text<const N: usize>(
    bytes: &[core::ffi::c_char; N],
    operation: &'static str,
) -> Result<String, ApiError> {
    // SAFETY: every shim text array is zero-filled before the runtime writes it.
    let text = unsafe { CStr::from_ptr(bytes.as_ptr()) };
    text.to_str().map(str::to_owned).map_err(|_| ApiError {
        operation,
        status: -1,
    })
}

#[cfg(fe2o3_hsa_runtime)]
fn measure_loaded_runtime_image(function_address: usize) -> Result<PayloadDigest, ApiError> {
    const MAX_RUNTIME_IMAGE_BYTES: u64 = 128 * 1024 * 1024;
    let maps = fs::read_to_string("/proc/self/maps").map_err(|_| ApiError {
        operation: "read /proc/self/maps for HSA runtime measurement",
        status: -1,
    })?;
    let mut mapping = None;
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let Some(range) = fields.next() else { continue };
        let Some((start, end)) = parse_address_range(range) else {
            continue;
        };
        if !(start..end).contains(&function_address) {
            continue;
        }
        let _permissions = fields.next();
        let _offset = fields.next();
        let device = fields.next().ok_or(ApiError {
            operation: "locate mapped runtime image device",
            status: -1,
        })?;
        let inode = fields
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value != 0)
            .ok_or(ApiError {
                operation: "locate mapped runtime image inode",
                status: -1,
            })?;
        let path = fields
            .next()
            .filter(|path| path.starts_with('/'))
            .ok_or(ApiError {
                operation: "locate mapped runtime image path",
                status: -1,
            })?;
        if fields.next().is_some() {
            return Err(ApiError {
                operation: "reject deleted or ambiguous mapped runtime image",
                status: -1,
            });
        }
        let (device_major, device_minor) = parse_device_number(device).ok_or(ApiError {
            operation: "parse mapped runtime image device",
            status: -1,
        })?;
        let candidate = (path.to_owned(), device_major, device_minor, inode);
        if mapping.replace(candidate).is_some() {
            return Err(ApiError {
                operation: "uniquely identify mapped runtime image",
                status: -1,
            });
        }
    }
    let (path, device_major, device_minor, inode) = mapping.ok_or(ApiError {
        operation: "uniquely identify mapped runtime image",
        status: -1,
    })?;
    let mut file = fs::File::open(path).map_err(|_| ApiError {
        operation: "open mapped runtime image",
        status: -1,
    })?;
    let before = file.metadata().map_err(|_| ApiError {
        operation: "stat opened runtime image",
        status: -1,
    })?;
    let (actual_major, actual_minor) = linux_device_components(before.dev());
    if !before.is_file()
        || before.ino() != inode
        || actual_major != device_major
        || actual_minor != device_minor
        || before.len() == 0
        || before.len() > MAX_RUNTIME_IMAGE_BYTES
    {
        return Err(ApiError {
            operation: "validate opened runtime image identity",
            status: -1,
        });
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.by_ref()
        .take(MAX_RUNTIME_IMAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ApiError {
            operation: "read opened runtime image",
            status: -1,
        })?;
    let after = file.metadata().map_err(|_| ApiError {
        operation: "restat opened runtime image",
        status: -1,
    })?;
    if bytes.len() as u64 != before.len()
        || before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return Err(ApiError {
            operation: "runtime image changed during measurement",
            status: -1,
        });
    }
    Ok(DigestAlgorithm::Sha256.calculate(&bytes))
}

#[cfg(fe2o3_hsa_runtime)]
fn parse_address_range(range: &str) -> Option<(usize, usize)> {
    let (start, end) = range.split_once('-')?;
    Some((
        usize::from_str_radix(start, 16).ok()?,
        usize::from_str_radix(end, 16).ok()?,
    ))
}

#[cfg(fe2o3_hsa_runtime)]
fn parse_device_number(device: &str) -> Option<(u64, u64)> {
    let (major, minor) = device.split_once(':')?;
    Some((
        u64::from_str_radix(major, 16).ok()?,
        u64::from_str_radix(minor, 16).ok()?,
    ))
}

#[cfg(fe2o3_hsa_runtime)]
const fn linux_device_components(device: u64) -> (u64, u64) {
    let major = ((device & 0x0000_0000_000f_ff00) >> 8) | ((device & 0xffff_f000_0000_0000) >> 32);
    let minor = (device & 0xff) | ((device & 0x0000_0fff_fff0_0000) >> 12);
    (major, minor)
}

fn derive_runtime_stack_digest(hsa: PayloadDigest, hip: PayloadDigest) -> PayloadDigest {
    let mut preimage = Vec::with_capacity(96);
    preimage.extend_from_slice(b"fe2o3-hsa-hip-runtime-stack-v1\0");
    preimage.extend_from_slice(hsa.bytes().as_bytes());
    preimage.extend_from_slice(hip.bytes().as_bytes());
    DigestAlgorithm::Sha256.calculate(&preimage)
}

#[cfg(fe2o3_hsa_runtime)]
fn derive_runtime_instance(
    image: PayloadDigest,
    hsa_function_address: usize,
    hip_function_address: usize,
) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"fe2o3-hsa-runtime-instance-v1\0");
    hasher.update(image.bytes().as_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(hsa_function_address.to_le_bytes());
    hasher.update(hip_function_address.to_le_bytes());
    let digest = hasher.finalize();
    let mut instance = [0; 16];
    instance.copy_from_slice(&digest[..16]);
    instance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_identity_zero_is_valid_but_missing_resources_are_not() {
        let record = sys::QueueRecord {
            pointer: 0x1000,
            id: 0,
            size: 64,
            async_error: 0x2000,
        };
        assert!(valid_queue_record(&record));
        for invalid in [
            sys::QueueRecord {
                pointer: 0,
                ..record
            },
            sys::QueueRecord { size: 0, ..record },
            sys::QueueRecord { size: 63, ..record },
            sys::QueueRecord {
                async_error: 0,
                ..record
            },
        ] {
            assert!(!valid_queue_record(&invalid));
        }
    }

    #[test]
    #[cfg(fe2o3_hsa_runtime)]
    fn malformed_queue_destroy_failure_retains_callback_authority() {
        let mut record = sys::QueueRecord {
            pointer: 0,
            id: u64::MAX,
            size: 0,
            async_error: 0,
        };
        // SAFETY: the native regression hook writes one complete test record.
        let status = unsafe { sys::fe2o3_hsa_test_malformed_queue_destroy_failure(&mut record) };
        assert_ne!(status, HSA_SUCCESS);
        assert_ne!(record.pointer, 0);
        assert_ne!(record.async_error, 0);
        assert_eq!(record.id, 0);
        assert_eq!(record.size, 3);

        // SAFETY: this hook releases only the fake allocations created above;
        // it never calls HSA and is not used by the production queue path.
        unsafe { sys::fe2o3_hsa_test_release_malformed_queue_record(&mut record) };
        assert_eq!(record.pointer, 0);
        assert_eq!(record.async_error, 0);
    }

    #[test]
    #[cfg(unix)]
    fn retained_malformed_queue_authority_is_terminal() {
        const CHILD: &str = "FE2O3_HSA_RETAINED_MALFORMED_QUEUE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let _ = queue_record_after_create(
                -1,
                sys::QueueRecord {
                    pointer: 0x1000,
                    id: 0,
                    size: 3,
                    async_error: 0x2000,
                },
            );
            std::process::exit(91);
        }

        use std::os::unix::process::ExitStatusExt;
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("api::tests::retained_malformed_queue_authority_is_terminal")
            .arg("--nocapture")
            .env(CHILD, "1");
        let status = crate::test_process_execution::status(&mut command).unwrap();
        assert_eq!(status.signal(), Some(6), "terminal queue status: {status}");
    }
}
