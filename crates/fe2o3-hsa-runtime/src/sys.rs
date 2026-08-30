#![allow(dead_code)]

use core::ffi::{c_char, c_int, c_void};

pub(crate) const AGENT_CAPACITY: usize = 64;
pub(crate) const POOL_CAPACITY: usize = 256;
pub(crate) const TEXT_CAPACITY: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct AgentRecord {
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
    pub name: [c_char; 64],
    pub uuid: [c_char; 22],
    pub isa: [c_char; TEXT_CAPACITY],
    pub matching_isa_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PoolRecord {
    pub handle: u64,
    pub owner_agent: u64,
    pub owner_node: u32,
    pub segment: u32,
    pub global_flags: u32,
    pub runtime_alloc_allowed: u32,
    pub runtime_alloc_alignment: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HipDeviceRecord {
    pub uuid: [u8; 16],
    pub pci_bus_id: [c_char; 32],
    pub round_trip_ordinal: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SymbolRecord {
    pub handle: u64,
    pub kernel_object: u64,
    pub kind: u32,
    pub kernarg_size: u32,
    pub kernarg_alignment: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub name: [c_char; TEXT_CAPACITY],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct QueueRecord {
    pub pointer: usize,
    pub id: u64,
    pub size: u32,
    pub async_error: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DispatchTimeRecord {
    pub start: u64,
    pub end: u64,
}

const _: () = {
    assert!(core::mem::size_of::<AgentRecord>() == 264);
    assert!(core::mem::align_of::<AgentRecord>() == 8);
    assert!(core::mem::size_of::<PoolRecord>() == 40);
    assert!(core::mem::align_of::<PoolRecord>() == 8);
    assert!(core::mem::size_of::<HipDeviceRecord>() == 52);
    assert!(core::mem::align_of::<HipDeviceRecord>() == 4);
    assert!(core::mem::size_of::<SymbolRecord>() == 168);
    assert!(core::mem::align_of::<SymbolRecord>() == 8);
    assert!(core::mem::size_of::<QueueRecord>() == 32);
    assert!(core::mem::align_of::<QueueRecord>() == 8);
    assert!(core::mem::size_of::<DispatchTimeRecord>() == 16);
    assert!(core::mem::align_of::<DispatchTimeRecord>() == 8);
};

#[cfg(fe2o3_hsa_runtime)]
unsafe extern "C" {
    pub fn fe2o3_hsa_init() -> c_int;
    pub fn fe2o3_hsa_shut_down() -> c_int;
    pub fn fe2o3_hsa_runtime_version(major: *mut u16, minor: *mut u16) -> c_int;
    pub fn fe2o3_hsa_runtime_function_address() -> usize;
    pub fn fe2o3_hip_runtime_function_address() -> usize;
    pub fn fe2o3_hsa_collect_agents(
        records: *mut AgentRecord,
        capacity: u32,
        count: *mut u32,
    ) -> c_int;
    pub fn fe2o3_hsa_collect_kernarg_pools(
        records: *mut PoolRecord,
        capacity: u32,
        count: *mut u32,
    ) -> c_int;
    pub fn fe2o3_hip_observe_device(ordinal: i32, record: *mut HipDeviceRecord) -> c_int;
    pub fn fe2o3_hsa_reader_create(bytes: *const c_void, len: usize, reader: *mut u64) -> c_int;
    pub fn fe2o3_hsa_reader_destroy(reader: u64) -> c_int;
    pub fn fe2o3_hsa_executable_create(profile: u32, executable: *mut u64) -> c_int;
    pub fn fe2o3_hsa_executable_load(
        executable: u64,
        agent: u64,
        reader: u64,
        loaded: *mut u64,
    ) -> c_int;
    pub fn fe2o3_hsa_executable_freeze(executable: u64) -> c_int;
    pub fn fe2o3_hsa_executable_destroy(executable: u64) -> c_int;
    pub fn fe2o3_hsa_resolve_symbol(
        executable: u64,
        agent: u64,
        name: *const c_char,
        record: *mut SymbolRecord,
    ) -> c_int;
    pub fn fe2o3_hsa_pool_allocate(pool: u64, len: usize, address: *mut *mut c_void) -> c_int;
    pub fn fe2o3_hsa_allow_access(agent: u64, address: *mut c_void) -> c_int;
    pub fn fe2o3_hsa_memory_free(address: *mut c_void) -> c_int;
    pub fn fe2o3_hsa_queue_create(agent: u64, size: u32, record: *mut QueueRecord) -> c_int;
    pub fn fe2o3_hsa_queue_async_error(record: *const QueueRecord) -> c_int;
    pub fn fe2o3_hsa_queue_enable_profiling(record: *const QueueRecord) -> c_int;
    pub fn fe2o3_hsa_queue_destroy(record: *mut QueueRecord) -> c_int;
    pub fn fe2o3_hsa_signal_create(initial_value: i64, signal: *mut u64) -> c_int;
    pub fn fe2o3_hsa_signal_destroy(signal: u64) -> c_int;
    pub fn fe2o3_hsa_signal_load_acquire(signal: u64) -> i64;
    pub fn fe2o3_hsa_signal_store_release(signal: u64, value: i64) -> c_int;
    pub fn fe2o3_hsa_system_timestamp_frequency(frequency: *mut u64) -> c_int;
    pub fn fe2o3_hsa_dispatch_time(
        agent: u64,
        signal: u64,
        record: *mut DispatchTimeRecord,
    ) -> c_int;
    #[cfg(test)]
    pub fn fe2o3_hsa_test_malformed_queue_destroy_failure(record: *mut QueueRecord) -> c_int;
    #[cfg(test)]
    pub fn fe2o3_hsa_test_release_malformed_queue_record(record: *mut QueueRecord);
    pub fn fe2o3_hsa_publish_kernel_dispatch(
        queue: *const QueueRecord,
        grid: *const u32,
        workgroup: *const u32,
        private_segment_size: u32,
        group_segment_size: u32,
        kernel_object: u64,
        kernarg: *mut c_void,
        completion_signal: u64,
        packet_id: *mut u64,
    ) -> c_int;
}
