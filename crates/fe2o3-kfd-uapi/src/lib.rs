#![no_std]
#![forbid(unsafe_code)]

//! Reviewed raw definitions for a deliberately small Linux KFD UAPI slice.
//!
//! This crate contains no file access, FFI, or `ioctl` execution. It provides
//! only C-layout data, request encodings, and fail-closed version admission for
//! a later syscall adapter. The admitted schema is pinned to KFD UAPI 1.18 as
//! shipped by the active AMDGPU 6.16.13 DKMS driver on the MI300X test host.

use core::mem::{align_of, offset_of, size_of};

/// Stable name of the frozen R1 discovery and identity UAPI schema.
pub const KFD_UAPI_SCHEMA_ID: &str = "linux-kfd-uapi-1.18-generic-ioc-v1";

/// Stable name of the reviewed R2 VM and memory-lifecycle UAPI extension.
pub const KFD_MEMORY_LIFECYCLE_SCHEMA_ID: &str = "linux-kfd-memory-lifecycle-1.18-generic-ioc-v1";

/// Stable name of the reviewed R4 compute-AQL queue-lifecycle UAPI extension.
pub const KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_ID: &str =
    "linux-kfd-aql-queue-lifecycle-1.18-generic-ioc-v1";

/// Path of the Linux UAPI header from which this schema was reviewed.
pub const KFD_UAPI_SOURCE_HEADER: &str = "include/uapi/linux/kfd_ioctl.h";

/// SHA-256 of the exact reviewed source header.
pub const KFD_UAPI_SOURCE_HEADER_SHA256: &str =
    "b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d";

/// SHA-256 of the active driver's SMI event stream implementation.
pub const KFD_UAPI_SMI_EVENTS_SOURCE_SHA256: &str =
    "2d786562fe1e97b8257841b755106c8bce47658a2aa3b439ce4e0178323004bd";

/// SHA-256 of the active driver's whole-GPU pre/post reset notification sites.
pub const KFD_UAPI_DEVICE_SOURCE_SHA256: &str =
    "ccf20227c5cdd5b258758f50f61bbc1008a09ea776c101f035f83963e7d23037";

/// SHA-256 of the active driver's KFD ioctl dispatch implementation.
pub const KFD_UAPI_CHARDEV_SOURCE_SHA256: &str =
    "f9a8805c5d479faee25e457051aa428e4bb523ecf1c7b1618a6a5f79ca5d7bba";

/// SHA-256 of the active driver's KFD GPUVM allocation and mapping implementation.
pub const KFD_UAPI_GPUVM_SOURCE_SHA256: &str =
    "c7cca2ee47a08c99bb73906662d82dd7d0b5738468fbef54848e5e6dd62ba50d";

/// SHA-256 of the active driver's queue-buffer acquisition implementation.
pub const KFD_AQL_QUEUE_BUFFER_SOURCE_SHA256: &str =
    "fb4b2a5c9e6981222873bcd7aca7e9c1397cba8f1a6b33634d2a48d4427fe062";

/// SHA-256 of the active driver's per-process queue lifecycle implementation.
pub const KFD_AQL_QUEUE_PQM_SOURCE_SHA256: &str =
    "8526e258824dbe145e4209cf0fed26463729234ba24369f39e3413e7e6e028db";

/// SHA-256 of the active driver's device queue manager implementation.
pub const KFD_AQL_QUEUE_DQM_SOURCE_SHA256: &str =
    "d61e53a78c1855c4badefbebb6c6ec52702be8cfe072253341c277337641c682";

/// SHA-256 of the active gfx9 MQD implementation used by gfx942.
pub const KFD_AQL_QUEUE_GFX9_MQD_SOURCE_SHA256: &str =
    "21166e9dbe2a4c24cbcd6f9ff6193aa093230e91fbafc8b4ac4eee1465cd2c9e";

/// SHA-256 of the active driver's internal queue and mmap definitions.
pub const KFD_AQL_QUEUE_PRIV_SOURCE_SHA256: &str =
    "f991330031c14725b2be0636ec1896ab530dc3d07d530ebd4f47efff97a82a99";

/// SHA-256 of the active gfx9 device queue manager implementation.
pub const KFD_AQL_QUEUE_DQM_GFX9_SOURCE_SHA256: &str =
    "53021a6f8211212f872545403e200d34d2e8c49b1cbdd17e382ae7baa43e52f2";

/// SHA-256 of the active HWS packet manager implementation.
pub const KFD_AQL_QUEUE_PACKET_MANAGER_SOURCE_SHA256: &str =
    "1ed642990cbb7d4cdbde211fee571318e233c19744ea1663d8eb68946c1310dd";

/// SHA-256 of the active kernel queue implementation used by packet management.
pub const KFD_AQL_QUEUE_KERNEL_QUEUE_SOURCE_SHA256: &str =
    "13e5d3634bcfed2ae871d8da0700cde47d8671eb014831b5d1ca95ed5a22fb36";

/// SHA-256 of the active MQD manager interface definitions.
pub const KFD_AQL_QUEUE_MQD_MANAGER_HEADER_SHA256: &str =
    "61ea7d4a13fb3168d0f026ecb13b13cf5846c86f233289043728b62ac9068605";

/// SHA-256 of the active device queue manager interface definitions.
pub const KFD_AQL_QUEUE_DQM_HEADER_SHA256: &str =
    "9e43b8f41ad89d1dd21fddf38dff4182f09b01218778f8278a743eacb72ceadd";

/// SHA-256 of the active gfx9 MQD structure definitions.
pub const KFD_AQL_QUEUE_V9_STRUCTS_HEADER_SHA256: &str =
    "18f8e59e4cab35d579d2e3f9fc4eadffd81d518d586065de4d9d0ab4fcc131d7";

/// SHA-256 of the active gfx9 KFD-to-KGD implementation.
pub const KFD_AQL_QUEUE_AMDGPU_GFX9_SOURCE_SHA256: &str =
    "d112169b3231439086da4943c7675bb4aeddb111b483a687fdd95794710ab27c";

/// SHA-256 of the active gfx9 KFD-to-KGD declarations.
pub const KFD_AQL_QUEUE_AMDGPU_GFX9_HEADER_SHA256: &str =
    "97bc6cd046c9c2495962d26d455e5231d95b0503385354177c366ea21fa9ed2e";

/// SHA-256 of the active generic gfx9 register offsets.
pub const KFD_AQL_QUEUE_GC9_OFFSET_HEADER_SHA256: &str =
    "dde287260e0b63eecfd7b723c1fdfaf9a3da7155f0ccd331385b9acc09433aa5";

/// SHA-256 of the active generic gfx9 register field definitions.
pub const KFD_AQL_QUEUE_GC9_SH_MASK_HEADER_SHA256: &str =
    "f67f3f753231a53e82e39783313605cd382eb9727f2cda775d6e849a7c38063e";

/// SHA-256 of the active gfx9.4.3 register field definitions used by gfx942.
pub const KFD_AQL_QUEUE_GC943_SH_MASK_HEADER_SHA256: &str =
    "8ee3fb2c721703a1643c118502e2900bd622b4d8d287103bd53922f92d35611b";

/// Canonical content manifest for the frozen R1 discovery and identity schema.
///
/// This identifies reviewed userspace definitions. It does not authenticate a
/// running kernel or claim that the driver implements the schema correctly.
pub const KFD_UAPI_SCHEMA_MANIFEST: &str = concat!(
    "schema_id=linux-kfd-uapi-1.18-generic-ioc-v1\n",
    "target=linux-x86_64-generic-ioc\n",
    "source_header=include/uapi/linux/kfd_ioctl.h\n",
    "source_header_sha256=b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d\n",
    "smi_events_source_sha256=2d786562fe1e97b8257841b755106c8bce47658a2aa3b439ce4e0178323004bd\n",
    "device_source_sha256=ccf20227c5cdd5b258758f50f61bbc1008a09ea776c101f035f83963e7d23037\n",
    "chardev_source_sha256=f9a8805c5d479faee25e457051aa428e4bb523ecf1c7b1618a6a5f79ca5d7bba\n",
    "source_package=amdgpu-dkms@1:6.16.13.30300400-2341068.24.04\n",
    "kfd_uapi=1.18\n",
    "get_version=size:8,align:4,major:0,minor:4,request:80084b01\n",
    "process_device_apertures=size:56,align:8,lds_base:0,lds_limit:8,scratch_base:16,scratch_limit:24,gpuvm_base:32,gpuvm_limit:40,gpu_id:48,pad:52\n",
    "get_process_apertures_new=size:16,align:8,process_apertures_ptr:0,num_of_nodes:8,pad:12,request:c0104b14\n",
    "acquire_vm=size:8,align:4,drm_fd:0,gpu_id:4,request:40084b15\n",
    "set_xnack_mode=size:4,align:4,xnack_enabled:0,request:c0044b21\n",
    "smi_events=size:8,align:4,gpu_id:0,anon_fd:4,request:c0084b1f,pre_reset:3,post_reset:4,mask:000000000000000c\n",
);

/// SHA-256 of [`KFD_UAPI_SCHEMA_MANIFEST`].
pub const KFD_UAPI_SCHEMA_MANIFEST_SHA256: &str =
    "e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4";

/// Typed digest bytes of [`KFD_UAPI_SCHEMA_MANIFEST`].
pub const KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES: [u8; 32] = [
    0xe4, 0xaa, 0xd5, 0xd8, 0xe3, 0x17, 0x7e, 0xa6, 0xd7, 0x02, 0x98, 0xad, 0xab, 0x77, 0x41, 0xc3,
    0x77, 0xcb, 0x09, 0x13, 0x73, 0x55, 0x3c, 0xe6, 0x89, 0xf3, 0x52, 0x5e, 0x75, 0x14, 0xd9, 0xb4,
];

/// Canonical manifest for the reviewed R2 VM and memory-lifecycle extension.
///
/// This is deliberately separate from KFD_UAPI_SCHEMA_MANIFEST. It binds
/// the frozen R1 schema digest as a prerequisite, then adds the active header
/// and GPUVM implementation provenance plus the exact reviewed memory ABI.
/// A future memory authority must bind both manifest digests; successful R1
/// version or device admission alone does not admit memory syscalls.
pub const KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST: &str = concat!(
    "schema_id=linux-kfd-memory-lifecycle-1.18-generic-ioc-v1\n",
    "base_schema_id=linux-kfd-uapi-1.18-generic-ioc-v1\n",
    "base_schema_manifest_sha256=e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4\n",
    "target=linux-x86_64-generic-ioc\n",
    "source_header=include/uapi/linux/kfd_ioctl.h\n",
    "source_header_sha256=b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d\n",
    "gpuvm_source=amd/amdgpu/amdgpu_amdkfd_gpuvm.c\n",
    "gpuvm_source_sha256=c7cca2ee47a08c99bb73906662d82dd7d0b5738468fbef54848e5e6dd62ba50d\n",
    "source_package=amdgpu-dkms@1:6.16.13.30300400-2341068.24.04\n",
    "kfd_uapi=1.18\n",
    "acquire_vm=size:8,align:4,drm_fd:0,gpu_id:4,request:40084b15\n",
    "alloc_flags=gtt:00000002,writable:80000000,executable:40000000,aql_queue:08000000,coherent:04000000,uncached:02000000\n",
    "alloc_profiles=host_visible_coherent:84000002,kernarg:86000002,aql_queue:8e000002,executable:c4000002\n",
    "alloc_memory=size:40,align:8,va_addr:0,size_field:8,handle:16,mmap_offset:24,gpu_id:32,flags:36,request:c0284b16\n",
    "free_memory=size:8,align:8,handle:0,request:40084b17\n",
    "map_memory=size:24,align:8,handle:0,device_ids_array_ptr:8,n_devices:16,n_success:20,request:c0184b18\n",
    "unmap_memory=size:24,align:8,handle:0,device_ids_array_ptr:8,n_devices:16,n_success:20,request:c0184b19\n",
);

/// SHA-256 of KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST.
pub const KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256: &str =
    "e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a";

/// Typed digest bytes of KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST.
pub const KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES: [u8; 32] = [
    0xe2, 0xd6, 0x98, 0x7b, 0x7c, 0x8e, 0x61, 0xa4, 0x05, 0xb2, 0xf7, 0x75, 0xd5, 0xd0, 0x04, 0xf4,
    0x58, 0xa0, 0x96, 0x24, 0x14, 0x59, 0xe4, 0xcf, 0xdf, 0x90, 0xbd, 0x44, 0x97, 0xf4, 0xd5, 0x8a,
];

/// Canonical manifest for the reviewed R4 compute-AQL queue UAPI extension.
///
/// Queue requests require the frozen R1 discovery schema and R2 memory schema,
/// but neither prerequisite authenticates this additional ABI or its driver
/// semantics. A future queue authority must bind this digest independently to
/// current kernel, process, device, allocation, and queue-lifecycle evidence.
/// The manifest pins the exact source set reviewed for gfx942 queue parsing,
/// lifecycle dispatch, scheduling packets, MQD programming, and register
/// definitions. It does not claim a complete transitive kernel build closure
/// or authenticate the code loaded by a running kernel.
pub const KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST: &str = concat!(
    "schema_id=linux-kfd-aql-queue-lifecycle-1.18-generic-ioc-v1\n",
    "base_schema_id=linux-kfd-uapi-1.18-generic-ioc-v1\n",
    "base_schema_manifest_sha256=e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4\n",
    "memory_schema_id=linux-kfd-memory-lifecycle-1.18-generic-ioc-v1\n",
    "memory_schema_manifest_sha256=e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a\n",
    "target=linux-x86_64-generic-ioc\n",
    "source_header=include/uapi/linux/kfd_ioctl.h\n",
    "source_header_sha256=b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d\n",
    "chardev_source=amd/amdkfd/kfd_chardev.c\n",
    "chardev_source_sha256=f9a8805c5d479faee25e457051aa428e4bb523ecf1c7b1618a6a5f79ca5d7bba\n",
    "queue_buffer_source=amd/amdkfd/kfd_queue.c\n",
    "queue_buffer_source_sha256=fb4b2a5c9e6981222873bcd7aca7e9c1397cba8f1a6b33634d2a48d4427fe062\n",
    "pqm_source=amd/amdkfd/kfd_process_queue_manager.c\n",
    "pqm_source_sha256=8526e258824dbe145e4209cf0fed26463729234ba24369f39e3413e7e6e028db\n",
    "dqm_source=amd/amdkfd/kfd_device_queue_manager.c\n",
    "dqm_source_sha256=d61e53a78c1855c4badefbebb6c6ec52702be8cfe072253341c277337641c682\n",
    "gfx9_mqd_source=amd/amdkfd/kfd_mqd_manager_v9.c\n",
    "gfx9_mqd_source_sha256=21166e9dbe2a4c24cbcd6f9ff6193aa093230e91fbafc8b4ac4eee1465cd2c9e\n",
    "queue_priv_source=amd/amdkfd/kfd_priv.h\n",
    "queue_priv_source_sha256=f991330031c14725b2be0636ec1896ab530dc3d07d530ebd4f47efff97a82a99\n",
    "gfx9_dqm_source=amd/amdkfd/kfd_device_queue_manager_v9.c\n",
    "gfx9_dqm_source_sha256=53021a6f8211212f872545403e200d34d2e8c49b1cbdd17e382ae7baa43e52f2\n",
    "packet_manager_source=amd/amdkfd/kfd_packet_manager.c\n",
    "packet_manager_source_sha256=1ed642990cbb7d4cdbde211fee571318e233c19744ea1663d8eb68946c1310dd\n",
    "kernel_queue_source=amd/amdkfd/kfd_kernel_queue.c\n",
    "kernel_queue_source_sha256=13e5d3634bcfed2ae871d8da0700cde47d8671eb014831b5d1ca95ed5a22fb36\n",
    "mqd_manager_header=amd/amdkfd/kfd_mqd_manager.h\n",
    "mqd_manager_header_sha256=61ea7d4a13fb3168d0f026ecb13b13cf5846c86f233289043728b62ac9068605\n",
    "dqm_header=amd/amdkfd/kfd_device_queue_manager.h\n",
    "dqm_header_sha256=9e43b8f41ad89d1dd21fddf38dff4182f09b01218778f8278a743eacb72ceadd\n",
    "v9_structs_header=amd/include/v9_structs.h\n",
    "v9_structs_header_sha256=18f8e59e4cab35d579d2e3f9fc4eadffd81d518d586065de4d9d0ab4fcc131d7\n",
    "amdgpu_gfx9_source=amd/amdgpu/amdgpu_amdkfd_gfx_v9.c\n",
    "amdgpu_gfx9_source_sha256=d112169b3231439086da4943c7675bb4aeddb111b483a687fdd95794710ab27c\n",
    "amdgpu_gfx9_header=amd/amdgpu/amdgpu_amdkfd_gfx_v9.h\n",
    "amdgpu_gfx9_header_sha256=97bc6cd046c9c2495962d26d455e5231d95b0503385354177c366ea21fa9ed2e\n",
    "gc9_offset_header=amd/include/asic_reg/gc/gc_9_0_offset.h\n",
    "gc9_offset_header_sha256=dde287260e0b63eecfd7b723c1fdfaf9a3da7155f0ccd331385b9acc09433aa5\n",
    "gc9_sh_mask_header=amd/include/asic_reg/gc/gc_9_0_sh_mask.h\n",
    "gc9_sh_mask_header_sha256=f67f3f753231a53e82e39783313605cd382eb9727f2cda775d6e849a7c38063e\n",
    "gc943_sh_mask_header=amd/include/asic_reg/gc/gc_9_4_3_sh_mask.h\n",
    "gc943_sh_mask_header_sha256=8ee3fb2c721703a1643c118502e2900bd622b4d8d287103bd53922f92d35611b\n",
    "source_package=amdgpu-dkms@1:6.16.13.30300400-2341068.24.04\n",
    "kfd_uapi=1.18\n",
    "semantic_gpu=gfx942\n",
    "semantic_source_set_scope=exact_reviewed_gfx942_queue_paths_v1_not_transitive_kernel_build_closure\n",
    "aql_profile=queue_type:00000002,percentage:0..100,priority:0..15,ring_size:power_of_two_and_at_least_1024,sdma_engine_id:0,pad:0\n",
    "update_profiles=reconfigure:ring_base_nonzero,disable:ring_base_zero_percentage_zero,size:power_of_two_and_at_least_1024,priority:0..15\n",
    "create_queue=size:96,align:8,ring_base:0,write_pointer:8,read_pointer:16,doorbell_offset:24,ring_size:32,gpu_id:36,queue_type:40,queue_percentage:44,queue_priority:48,queue_id:52,eop_address:56,eop_size:64,ctx_address:72,ctx_size:80,ctl_stack_size:84,sdma_engine_id:88,pad:92,request:c0604b02\n",
    "destroy_queue=size:8,align:4,queue_id:0,pad:4,request:c0084b03\n",
    "update_queue=size:24,align:8,ring_base:0,queue_id:8,ring_size:12,queue_percentage:16,queue_priority:20,request:40184b07\n",
);

/// SHA-256 of [`KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST`].
pub const KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256: &str =
    "b11f3c8c766dd25394350646e35269e10c8a33acb98f74cba2a82e95fa185c4e";

/// Typed digest bytes of [`KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST`].
pub const KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES: [u8; 32] = [
    0xb1, 0x1f, 0x3c, 0x8c, 0x76, 0x6d, 0xd2, 0x53, 0x94, 0x35, 0x06, 0x46, 0xe3, 0x52, 0x69, 0xe1,
    0x0c, 0x8a, 0x33, 0xac, 0xb9, 0x8f, 0x74, 0xcb, 0xa2, 0xa8, 0x2e, 0x95, 0xfa, 0x18, 0x5c, 0x4e,
];

/// Major version declared by the reviewed AMDGPU 6.16.13 KFD UAPI header.
pub const KFD_IOCTL_MAJOR_VERSION: u32 = 1;

/// Minor version declared by the reviewed AMDGPU 6.16.13 KFD UAPI header.
pub const KFD_IOCTL_MINOR_VERSION: u32 = 18;

/// The only minor version admitted by this initial schema.
pub const KFD_IOCTL_MIN_ADMITTED_MINOR_VERSION: u32 = KFD_IOCTL_MINOR_VERSION;

/// The newest minor version reviewed by this initial schema.
pub const KFD_IOCTL_MAX_ADMITTED_MINOR_VERSION: u32 = KFD_IOCTL_MINOR_VERSION;

/// The KFD ioctl type byte (`'K'`).
pub const AMDKFD_IOCTL_BASE: u8 = b'K';

/// GTT/system-memory allocation type admitted by this schema.
pub const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;

/// Permit GPU writes to the allocation.
pub const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;

/// Permit GPU instruction fetches from the allocation.
pub const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;

/// Request the KFD AQL queue double-mapping behavior.
pub const KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;

/// Request coherent backing memory.
pub const KFD_IOC_ALLOC_MEM_FLAGS_COHERENT: u32 = 1 << 26;

/// Request uncached backing memory.
pub const KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED: u32 = 1 << 25;

/// Exact admitted profile for ordinary host-visible coherent memory.
pub const KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT: u32 = KFD_IOC_ALLOC_MEM_FLAGS_GTT
    | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
    | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

/// Exact admitted profile for host-visible kernarg memory.
pub const KFD_ALLOC_MEMORY_FLAGS_KERNARG: u32 =
    KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;

/// Exact admitted profile for an AQL queue ring's double-mapped storage.
pub const KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE: u32 =
    KFD_ALLOC_MEMORY_FLAGS_KERNARG | KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM;

/// Exact admitted profile for host-visible executable memory.
pub const KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE: u32 =
    KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

/// Exact UAPI queue type admitted by the R4 builder.
pub const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;

/// Maximum low-byte queue percentage accepted by the active driver.
pub const KFD_MAX_QUEUE_PERCENTAGE: u32 = 100;

/// Maximum relative queue priority accepted by the active driver.
pub const KFD_MAX_QUEUE_PRIORITY: u32 = 15;

/// Minimum queue ring size declared by KFD UAPI 1.18.
pub const KFD_MIN_QUEUE_RING_SIZE: u32 = 1024;

/// An exact, reviewed allocation-flag profile accepted by fe2o3's R2 builder.
///
/// The private field prevents callers from constructing novel bit
/// combinations through the typed builder. Raw UAPI records remain inspectable
/// and serializable as C-layout data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KfdAllocMemoryFlags(u32);

impl KfdAllocMemoryFlags {
    pub const HOST_VISIBLE_COHERENT: Self = Self(KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT);
    pub const KERNARG: Self = Self(KFD_ALLOC_MEMORY_FLAGS_KERNARG);
    pub const AQL_QUEUE: Self = Self(KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE);
    pub const EXECUTABLE: Self = Self(KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE);

    /// Returns the exact KFD UAPI bit pattern carried on the wire.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Why an allocation-flag bit pattern was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdAllocMemoryFlagsError {
    /// The pattern is not one of the four exact reviewed profiles.
    Unsupported { flags: u32 },
}

/// Admits only the exact host-visible profiles reviewed for R2.
///
/// In particular, this rejects every pattern containing VRAM, USERPTR/SVM,
/// doorbell, MMIO, public, or unknown/reserved flag bits. Peer mapping is a
/// separate adapter-level device-array policy.
pub const fn admit_kfd_alloc_memory_flags(
    flags: u32,
) -> Result<KfdAllocMemoryFlags, KfdAllocMemoryFlagsError> {
    match flags {
        KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT => {
            Ok(KfdAllocMemoryFlags::HOST_VISIBLE_COHERENT)
        }
        KFD_ALLOC_MEMORY_FLAGS_KERNARG => Ok(KfdAllocMemoryFlags::KERNARG),
        KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE => Ok(KfdAllocMemoryFlags::AQL_QUEUE),
        KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE => Ok(KfdAllocMemoryFlags::EXECUTABLE),
        flags => Err(KfdAllocMemoryFlagsError::Unsupported { flags }),
    }
}

/// A power-of-two AQL ring size at least [`KFD_MIN_QUEUE_RING_SIZE`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KfdAqlQueueRingSize(u32);

impl KfdAqlQueueRingSize {
    pub const fn bytes(self) -> u32 {
        self.0
    }
}

/// A nonzero numeric ring address observation for UPDATE_QUEUE.
///
/// This type proves only that the integer is nonzero. It does not prove pointer
/// provenance, alignment, mapped length, GPU visibility, allocation ownership,
/// or that the address belongs to the queue identified by the request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KfdAqlQueueRingAddress(u64);

impl KfdAqlQueueRingAddress {
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A zero numeric address cannot be used for queue reconfiguration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdAqlQueueRingAddressError;

/// Admits a nonzero numeric ring address observation without making a
/// provenance or ownership claim.
pub const fn admit_kfd_aql_queue_ring_address(
    address: u64,
) -> Result<KfdAqlQueueRingAddress, KfdAqlQueueRingAddressError> {
    if address == 0 {
        return Err(KfdAqlQueueRingAddressError);
    }
    Ok(KfdAqlQueueRingAddress(address))
}

/// Why an AQL ring size was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdAqlQueueRingSizeError {
    BelowMinimum { size: u32, minimum: u32 },
    NotPowerOfTwo { size: u32 },
}

/// Admits the exact ring-size range the active queue parser accepts without
/// clamping or normalizing the caller's input.
pub const fn admit_kfd_aql_queue_ring_size(
    size: u32,
) -> Result<KfdAqlQueueRingSize, KfdAqlQueueRingSizeError> {
    if size < KFD_MIN_QUEUE_RING_SIZE {
        return Err(KfdAqlQueueRingSizeError::BelowMinimum {
            size,
            minimum: KFD_MIN_QUEUE_RING_SIZE,
        });
    }
    if !size.is_power_of_two() {
        return Err(KfdAqlQueueRingSizeError::NotPowerOfTwo { size });
    }
    Ok(KfdAqlQueueRingSize(size))
}

/// A queue activity percentage with no repurposed target-XCC bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KfdQueuePercentage(u32);

impl KfdQueuePercentage {
    /// Exact inactive percentage used by the reviewed disable builder.
    pub const DISABLED: Self = Self(0);

    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Why a queue percentage was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdQueuePercentageError {
    pub percentage: u32,
    pub maximum: u32,
}

/// Admits only the unextended 0 through 100 queue-percentage field.
///
/// The active driver repurposes bits 8 through 15 as a PM4 target-XCC selector.
/// This initial compute-AQL profile rejects that extension by accepting only
/// the ordinary percentage range.
pub const fn admit_kfd_queue_percentage(
    percentage: u32,
) -> Result<KfdQueuePercentage, KfdQueuePercentageError> {
    if percentage > KFD_MAX_QUEUE_PERCENTAGE {
        return Err(KfdQueuePercentageError {
            percentage,
            maximum: KFD_MAX_QUEUE_PERCENTAGE,
        });
    }
    Ok(KfdQueuePercentage(percentage))
}

/// A relative queue priority in the KFD UAPI 1.18 range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KfdQueuePriority(u32);

impl KfdQueuePriority {
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Why a queue priority was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdQueuePriorityError {
    pub priority: u32,
    pub maximum: u32,
}

/// Admits the complete relative-priority range declared by KFD UAPI 1.18.
pub const fn admit_kfd_queue_priority(
    priority: u32,
) -> Result<KfdQueuePriority, KfdQueuePriorityError> {
    if priority > KFD_MAX_QUEUE_PRIORITY {
        return Err(KfdQueuePriorityError {
            priority,
            maximum: KFD_MAX_QUEUE_PRIORITY,
        });
    }
    Ok(KfdQueuePriority(priority))
}

/// Negative input queries the current process XNACK mode without changing it.
pub const KFD_XNACK_MODE_QUERY: i32 = -1;

/// Input and normalized output value for disabled process XNACK mode.
pub const KFD_XNACK_MODE_DISABLED: i32 = 0;

/// Canonical positive input value for enabled process XNACK mode.
pub const KFD_XNACK_MODE_ENABLED: i32 = 1;

/// KFD SMI event index emitted before a whole-GPU reset.
pub const KFD_SMI_EVENT_GPU_PRE_RESET: u32 = 3;

/// KFD SMI event index emitted after a whole-GPU reset.
pub const KFD_SMI_EVENT_GPU_POST_RESET: u32 = 4;

/// Event mask enabling only whole-GPU pre/post reset notifications.
pub const KFD_SMI_EVENT_GPU_RESET_MASK: u64 =
    (1_u64 << (KFD_SMI_EVENT_GPU_PRE_RESET - 1)) | (1_u64 << (KFD_SMI_EVENT_GPU_POST_RESET - 1));

/// Maximum single SMI event message size declared by KFD UAPI 1.18.
pub const KFD_SMI_EVENT_MSG_SIZE: usize = 96;

/// Linux generic ioctl request number type.
pub type IoctlRequest = u32;

const IOC_NR_BITS: u32 = 8;
const IOC_TYPE_BITS: u32 = 8;
const IOC_SIZE_BITS: u32 = 14;
const IOC_NR_SHIFT: u32 = 0;
const IOC_TYPE_SHIFT: u32 = IOC_NR_SHIFT + IOC_NR_BITS;
const IOC_SIZE_SHIFT: u32 = IOC_TYPE_SHIFT + IOC_TYPE_BITS;
const IOC_DIR_SHIFT: u32 = IOC_SIZE_SHIFT + IOC_SIZE_BITS;
const IOC_SIZE_MASK: usize = (1usize << IOC_SIZE_BITS) - 1;

/// Transfer direction encoded by Linux's generic `_IOC` convention.
///
/// `Write` means userspace writes data that the kernel reads. `Read` means the
/// kernel writes data that userspace reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum IoctlDirection {
    None = 0,
    Write = 1,
    Read = 2,
    ReadWrite = 3,
}

/// Encodes a Linux generic ioctl request without libc or generated bindings.
///
/// `None` is returned if the payload does not fit the generic 14-bit size
/// field. This helper models the generic Linux encoding used by the admitted
/// x86_64 runtime target; an adapter for an architecture that overrides `_IOC`
/// must define and review a separate schema.
pub const fn encode_ioctl(
    direction: IoctlDirection,
    ioctl_type: u8,
    number: u8,
    payload_size: usize,
) -> Option<IoctlRequest> {
    if payload_size > IOC_SIZE_MASK {
        return None;
    }

    Some(
        ((direction as u32) << IOC_DIR_SHIFT)
            | ((ioctl_type as u32) << IOC_TYPE_SHIFT)
            | ((number as u32) << IOC_NR_SHIFT)
            | ((payload_size as u32) << IOC_SIZE_SHIFT),
    )
}

const fn encode_admitted_ioctl(
    direction: IoctlDirection,
    ioctl_type: u8,
    number: u8,
    payload_size: usize,
) -> IoctlRequest {
    match encode_ioctl(direction, ioctl_type, number, payload_size) {
        Some(request) => request,
        None => panic!("admitted KFD ioctl payload exceeds Linux _IOC size field"),
    }
}

/// C layout of `struct kfd_ioctl_get_version_args`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlGetVersionArgs {
    /// KFD UAPI major version returned by the kernel.
    pub major_version: u32,
    /// KFD UAPI minor version returned by the kernel.
    pub minor_version: u32,
}

impl KfdIoctlGetVersionArgs {
    /// Creates a zero-initialized output buffer for `AMDKFD_IOC_GET_VERSION`.
    pub const fn zeroed() -> Self {
        Self {
            major_version: 0,
            minor_version: 0,
        }
    }

    /// Converts the raw output into the value consumed by version admission.
    pub const fn reported_version(self) -> KfdUapiVersion {
        KfdUapiVersion {
            major: self.major_version,
            minor: self.minor_version,
        }
    }
}

/// Opaque userspace addresses and device-derived auxiliary sizes for a
/// compute-AQL queue creation request.
///
/// This data-only record does not validate address provenance, mapped length,
/// alignment, GPU visibility, EOP size, CWSR size, or control-stack size. Those
/// facts depend on admitted allocations and selected-device topology and must
/// be established by the future queue adapter before issuing a request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdAqlComputeQueueBuffers {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
}

/// C layout of `struct kfd_ioctl_create_queue_args`.
///
/// The safe constructor fixes the queue kind to compute AQL, zeros SDMA and
/// padding inputs, and initializes both kernel outputs to fail-closed
/// sentinels. It does not create a queue or grant authority over any address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlCreateQueueArgs {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,
    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,
    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub sdma_engine_id: u32,
    pub pad: u32,
}

impl KfdIoctlCreateQueueArgs {
    /// Builds only the reviewed initial compute-AQL input profile.
    pub const fn new_compute_aql(
        buffers: KfdAqlComputeQueueBuffers,
        ring_size: KfdAqlQueueRingSize,
        gpu_id: u32,
        queue_percentage: KfdQueuePercentage,
        queue_priority: KfdQueuePriority,
    ) -> Self {
        Self {
            ring_base_address: buffers.ring_base_address,
            write_pointer_address: buffers.write_pointer_address,
            read_pointer_address: buffers.read_pointer_address,
            doorbell_offset: u64::MAX,
            ring_size: ring_size.bytes(),
            gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: queue_percentage.value(),
            queue_priority: queue_priority.value(),
            queue_id: u32::MAX,
            eop_buffer_address: buffers.eop_buffer_address,
            eop_buffer_size: buffers.eop_buffer_size,
            ctx_save_restore_address: buffers.ctx_save_restore_address,
            ctx_save_restore_size: buffers.ctx_save_restore_size,
            ctl_stack_size: buffers.ctl_stack_size,
            sdma_engine_id: 0,
            pad: 0,
        }
    }
}

/// C layout of `struct kfd_ioctl_destroy_queue_args`.
///
/// A numeric queue ID is not queue authority. A future adapter must prove that
/// the ID is live, process-owned, current, and not already being destroyed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlDestroyQueueArgs {
    pub queue_id: u32,
    pub pad: u32,
}

impl KfdIoctlDestroyQueueArgs {
    pub const fn new(queue_id: u32) -> Self {
        Self { queue_id, pad: 0 }
    }
}

/// C layout of `struct kfd_ioctl_update_queue_args`.
///
/// The UAPI record does not carry queue format. This constructor's AQL name
/// describes the admitted inputs only; a future adapter must bind `queue_id`
/// to a live compute-AQL queue before issuing the request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlUpdateQueueArgs {
    pub ring_base_address: u64,
    pub queue_id: u32,
    pub ring_size: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
}

impl KfdIoctlUpdateQueueArgs {
    /// Reconfigures a queue with a deliberately nonzero numeric ring address.
    ///
    /// The address observation is not allocation or queue authority. The
    /// adapter must still prove the ring is mapped, sufficiently large, and
    /// owned by the same live compute-AQL queue.
    pub const fn reconfigure_compute_aql(
        queue_id: u32,
        ring_base_address: KfdAqlQueueRingAddress,
        ring_size: KfdAqlQueueRingSize,
        queue_percentage: KfdQueuePercentage,
        queue_priority: KfdQueuePriority,
    ) -> Self {
        Self {
            ring_base_address: ring_base_address.value(),
            queue_id,
            ring_size: ring_size.bytes(),
            queue_percentage: queue_percentage.value(),
            queue_priority: queue_priority.value(),
        }
    }

    /// Encodes the active driver's deliberate NULL-ring disable operation.
    ///
    /// The driver treats a zero ring address as disabled, while still parsing
    /// and storing ring size and priority and updating the MQD. Therefore this
    /// builder requires an admitted nonzero size, fixes percentage to zero,
    /// and admits priority normally. It does not prove queue ownership or that
    /// a failed ioctl left the queue active or inactive.
    pub const fn disable_compute_aql(
        queue_id: u32,
        ring_size: KfdAqlQueueRingSize,
        queue_priority: KfdQueuePriority,
    ) -> Self {
        Self {
            ring_base_address: 0,
            queue_id,
            ring_size: ring_size.bytes(),
            queue_percentage: KfdQueuePercentage::DISABLED.value(),
            queue_priority: queue_priority.value(),
        }
    }
}

/// C layout of `struct kfd_ioctl_acquire_vm_args`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlAcquireVmArgs {
    /// Nonnegative DRM render-node file descriptor, represented by the UAPI as `__u32`.
    pub drm_fd: u32,
    /// KFD topology GPU identifier whose VM is being acquired.
    pub gpu_id: u32,
}

impl KfdIoctlAcquireVmArgs {
    /// Constructs the raw request after a higher layer validates descriptor and device identity.
    pub const fn new(drm_fd: u32, gpu_id: u32) -> Self {
        Self { drm_fd, gpu_id }
    }
}

/// C layout of `struct kfd_ioctl_alloc_memory_of_gpu_args`.
///
/// `va_addr`, the returned `handle`, and the returned `mmap_offset` are opaque
/// integer values. This data-only crate neither dereferences them nor assigns
/// ownership. USERPTR is not admitted, so the input `mmap_offset` is always
/// initialized to zero by [`Self::new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlAllocMemoryOfGpuArgs {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

impl KfdIoctlAllocMemoryOfGpuArgs {
    /// Constructs an allocation request from an exact admitted profile.
    ///
    /// Kernel-output fields are zeroed so stale handles and mmap offsets cannot
    /// cross the syscall boundary if a later adapter reuses caller storage.
    pub const fn new(va_addr: u64, size: u64, gpu_id: u32, flags: KfdAllocMemoryFlags) -> Self {
        Self {
            va_addr,
            size,
            handle: 0,
            mmap_offset: 0,
            gpu_id,
            flags: flags.bits(),
        }
    }
}

/// C layout of `struct kfd_ioctl_free_memory_of_gpu_args`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlFreeMemoryOfGpuArgs {
    /// Opaque KFD allocation handle returned by the allocation request.
    pub handle: u64,
}

impl KfdIoctlFreeMemoryOfGpuArgs {
    pub const fn new(handle: u64) -> Self {
        Self { handle }
    }
}

/// C layout of `struct kfd_ioctl_map_memory_to_gpu_args`.
///
/// `device_ids_array_ptr` is an opaque userspace address. A syscall adapter
/// must retain and bound the backing `u32` array for the full ioctl. The
/// adapter, not this record, is also responsible for rejecting peer devices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlMapMemoryToGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    /// In/out prefix length. Preserve the kernel-written value after failure.
    pub n_success: u32,
}

impl KfdIoctlMapMemoryToGpuArgs {
    /// Constructs a first-attempt request with no completed prefix.
    pub const fn initial(handle: u64, device_ids_array_ptr: u64, n_devices: u32) -> Self {
        Self {
            handle,
            device_ids_array_ptr,
            n_devices,
            n_success: 0,
        }
    }

    /// Constructs a retry while preserving the exact completed prefix.
    ///
    /// Semantic validation such as `n_success <= n_devices` remains with the
    /// syscall adapter so this raw record never silently clamps kernel state.
    pub const fn retry(
        handle: u64,
        device_ids_array_ptr: u64,
        n_devices: u32,
        n_success: u32,
    ) -> Self {
        Self {
            handle,
            device_ids_array_ptr,
            n_devices,
            n_success,
        }
    }
}

/// C layout of `struct kfd_ioctl_unmap_memory_from_gpu_args`.
///
/// Field semantics and buffer-lifetime requirements match
/// [`KfdIoctlMapMemoryToGpuArgs`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlUnmapMemoryFromGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    /// In/out prefix length. Preserve the kernel-written value after failure.
    pub n_success: u32,
}

impl KfdIoctlUnmapMemoryFromGpuArgs {
    /// Constructs a first-attempt request with no completed prefix.
    pub const fn initial(handle: u64, device_ids_array_ptr: u64, n_devices: u32) -> Self {
        Self {
            handle,
            device_ids_array_ptr,
            n_devices,
            n_success: 0,
        }
    }

    /// Constructs a retry while preserving the exact completed prefix.
    pub const fn retry(
        handle: u64,
        device_ids_array_ptr: u64,
        n_devices: u32,
        n_success: u32,
    ) -> Self {
        Self {
            handle,
            device_ids_array_ptr,
            n_devices,
            n_success,
        }
    }
}

/// C layout of `struct kfd_process_device_apertures`.
///
/// The record reports one process-visible virtual address aperture set and its
/// KFD `gpu_id`. It does not allocate, map, or grant authority over any range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct KfdProcessDeviceApertures {
    pub lds_base: u64,
    pub lds_limit: u64,
    pub scratch_base: u64,
    pub scratch_limit: u64,
    pub gpuvm_base: u64,
    pub gpuvm_limit: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

/// C layout of `struct kfd_ioctl_get_process_apertures_new_args`.
///
/// `kfd_process_device_apertures_ptr` is an opaque userspace address. A later
/// syscall adapter owns its provenance, capacity, lifetime, and initialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlGetProcessAperturesNewArgs {
    pub kfd_process_device_apertures_ptr: u64,
    pub num_of_nodes: u32,
    pub pad: u32,
}

impl KfdIoctlGetProcessAperturesNewArgs {
    /// Constructs an argument record for a caller-owned output array.
    pub const fn new(kfd_process_device_apertures_ptr: u64, capacity: u32) -> Self {
        Self {
            kfd_process_device_apertures_ptr,
            num_of_nodes: capacity,
            pad: 0,
        }
    }
}

/// C layout of `struct kfd_ioctl_set_xnack_mode_args`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlSetXnackModeArgs {
    /// Negative queries, zero disables, and positive enables process XNACK mode.
    pub xnack_enabled: i32,
}

impl KfdIoctlSetXnackModeArgs {
    /// Constructs a query that leaves the process mode unchanged.
    pub const fn query() -> Self {
        Self {
            xnack_enabled: KFD_XNACK_MODE_QUERY,
        }
    }

    /// Constructs a request for one canonical process mode.
    pub const fn set(enabled: bool) -> Self {
        Self {
            xnack_enabled: if enabled {
                KFD_XNACK_MODE_ENABLED
            } else {
                KFD_XNACK_MODE_DISABLED
            },
        }
    }
}

/// C layout of `struct kfd_ioctl_smi_events_args`.
///
/// A successful request returns an anonymous event descriptor in `anon_fd`.
/// Descriptor ownership, event-mask writes, reads, and close-on-exec policy
/// belong to the syscall adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KfdIoctlSmiEventsArgs {
    pub gpu_id: u32,
    pub anon_fd: u32,
}

impl KfdIoctlSmiEventsArgs {
    /// Constructs a request with a fail-closed sentinel output descriptor.
    pub const fn new(gpu_id: u32) -> Self {
        Self {
            gpu_id,
            anon_fd: u32::MAX,
        }
    }
}

/// Request number for `_IOR('K', 0x01, struct kfd_ioctl_get_version_args)`.
pub const AMDKFD_IOC_GET_VERSION: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::Read,
    AMDKFD_IOCTL_BASE,
    0x01,
    size_of::<KfdIoctlGetVersionArgs>(),
);

/// Request for `_IOWR('K', 0x02, struct kfd_ioctl_create_queue_args)`.
///
/// This constant is intentionally not exposed through [`AdmittedKfdUapi`]: a
/// successful version query does not authenticate the separate queue schema.
pub const AMDKFD_IOC_CREATE_QUEUE: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x02,
    size_of::<KfdIoctlCreateQueueArgs>(),
);

/// Request for `_IOWR('K', 0x03, struct kfd_ioctl_destroy_queue_args)`.
pub const AMDKFD_IOC_DESTROY_QUEUE: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x03,
    size_of::<KfdIoctlDestroyQueueArgs>(),
);

/// Request for `_IOW('K', 0x07, struct kfd_ioctl_update_queue_args)`.
pub const AMDKFD_IOC_UPDATE_QUEUE: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::Write,
    AMDKFD_IOCTL_BASE,
    0x07,
    size_of::<KfdIoctlUpdateQueueArgs>(),
);

/// Request number for `_IOW('K', 0x15, struct kfd_ioctl_acquire_vm_args)`.
pub const AMDKFD_IOC_ACQUIRE_VM: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::Write,
    AMDKFD_IOCTL_BASE,
    0x15,
    size_of::<KfdIoctlAcquireVmArgs>(),
);

/// Request for `_IOWR('K', 0x16, struct kfd_ioctl_alloc_memory_of_gpu_args)`.
pub const AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x16,
    size_of::<KfdIoctlAllocMemoryOfGpuArgs>(),
);

/// Request for `_IOW('K', 0x17, struct kfd_ioctl_free_memory_of_gpu_args)`.
pub const AMDKFD_IOC_FREE_MEMORY_OF_GPU: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::Write,
    AMDKFD_IOCTL_BASE,
    0x17,
    size_of::<KfdIoctlFreeMemoryOfGpuArgs>(),
);

/// Request for `_IOWR('K', 0x18, struct kfd_ioctl_map_memory_to_gpu_args)`.
pub const AMDKFD_IOC_MAP_MEMORY_TO_GPU: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x18,
    size_of::<KfdIoctlMapMemoryToGpuArgs>(),
);

/// Request for `_IOWR('K', 0x19, struct kfd_ioctl_unmap_memory_from_gpu_args)`.
pub const AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x19,
    size_of::<KfdIoctlUnmapMemoryFromGpuArgs>(),
);

/// Request for `_IOWR('K', 0x14, struct kfd_ioctl_get_process_apertures_new_args)`.
pub const AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x14,
    size_of::<KfdIoctlGetProcessAperturesNewArgs>(),
);

/// Request number for `_IOWR('K', 0x21, struct kfd_ioctl_set_xnack_mode_args)`.
pub const AMDKFD_IOC_SET_XNACK_MODE: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x21,
    size_of::<KfdIoctlSetXnackModeArgs>(),
);

/// Request for `_IOWR('K', 0x1f, struct kfd_ioctl_smi_events_args)`.
pub const AMDKFD_IOC_SMI_EVENTS: IoctlRequest = encode_admitted_ioctl(
    IoctlDirection::ReadWrite,
    AMDKFD_IOCTL_BASE,
    0x1f,
    size_of::<KfdIoctlSmiEventsArgs>(),
);

/// KFD UAPI version reported by `AMDKFD_IOC_GET_VERSION`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KfdUapiVersion {
    pub major: u32,
    pub minor: u32,
}

impl KfdUapiVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

/// Evidence that a reported KFD version is covered by this reviewed schema.
///
/// The private field prevents callers from constructing admission evidence
/// without passing [`negotiate_kfd_uapi_version`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmittedKfdUapi {
    reported: KfdUapiVersion,
}

impl AdmittedKfdUapi {
    pub const fn reported_version(self) -> KfdUapiVersion {
        self.reported
    }

    pub const fn schema_id(self) -> &'static str {
        KFD_UAPI_SCHEMA_ID
    }

    pub const fn schema_manifest_sha256(self) -> &'static str {
        KFD_UAPI_SCHEMA_MANIFEST_SHA256
    }

    /// Returns the admitted ACQUIRE_VM request number.
    ///
    /// Keeping this method on the admission token lets higher-level adapters
    /// require reviewed version evidence before exposing the operation.
    pub const fn acquire_vm_request(self) -> IoctlRequest {
        AMDKFD_IOC_ACQUIRE_VM
    }

    /// Returns the admitted GET_PROCESS_APERTURES_NEW request number.
    pub const fn get_process_apertures_new_request(self) -> IoctlRequest {
        AMDKFD_IOC_GET_PROCESS_APERTURES_NEW
    }

    /// Returns the admitted SET_XNACK_MODE request number.
    pub const fn set_xnack_mode_request(self) -> IoctlRequest {
        AMDKFD_IOC_SET_XNACK_MODE
    }

    /// Returns the admitted SMI_EVENTS request number.
    pub const fn smi_events_request(self) -> IoctlRequest {
        AMDKFD_IOC_SMI_EVENTS
    }
}

/// Why a kernel-reported KFD UAPI version was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdUapiVersionError {
    UnsupportedMajor { reported: u32, admitted: u32 },
    MinorTooOld { reported: u32, minimum: u32 },
    MinorNewerThanReviewed { reported: u32, maximum: u32 },
}

/// Admits only versions whose semantics and layout were explicitly reviewed.
///
/// This initial foundation intentionally accepts exactly KFD UAPI 1.18. A
/// newer minor version may be backwards compatible in Linux, but fe2o3 must
/// review it and extend this schema before making a formal compatibility claim.
pub const fn negotiate_kfd_uapi_version(
    reported: KfdUapiVersion,
) -> Result<AdmittedKfdUapi, KfdUapiVersionError> {
    if reported.major != KFD_IOCTL_MAJOR_VERSION {
        return Err(KfdUapiVersionError::UnsupportedMajor {
            reported: reported.major,
            admitted: KFD_IOCTL_MAJOR_VERSION,
        });
    }
    if reported.minor < KFD_IOCTL_MIN_ADMITTED_MINOR_VERSION {
        return Err(KfdUapiVersionError::MinorTooOld {
            reported: reported.minor,
            minimum: KFD_IOCTL_MIN_ADMITTED_MINOR_VERSION,
        });
    }
    if reported.minor > KFD_IOCTL_MAX_ADMITTED_MINOR_VERSION {
        return Err(KfdUapiVersionError::MinorNewerThanReviewed {
            reported: reported.minor,
            maximum: KFD_IOCTL_MAX_ADMITTED_MINOR_VERSION,
        });
    }

    Ok(AdmittedKfdUapi { reported })
}

// Compile-time ABI assertions for the admitted Linux KFD 1.18 schema.
const _: () = {
    assert!(size_of::<KfdIoctlGetVersionArgs>() == 8);
    assert!(align_of::<KfdIoctlGetVersionArgs>() == 4);
    assert!(offset_of!(KfdIoctlGetVersionArgs, major_version) == 0);
    assert!(offset_of!(KfdIoctlGetVersionArgs, minor_version) == 4);

    assert!(size_of::<KfdIoctlCreateQueueArgs>() == 96);
    assert!(align_of::<KfdIoctlCreateQueueArgs>() == 8);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, ring_base_address) == 0);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, write_pointer_address) == 8);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, read_pointer_address) == 16);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, doorbell_offset) == 24);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, ring_size) == 32);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, gpu_id) == 36);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, queue_type) == 40);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, queue_percentage) == 44);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, queue_priority) == 48);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, queue_id) == 52);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, eop_buffer_address) == 56);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, eop_buffer_size) == 64);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, ctx_save_restore_address) == 72);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, ctx_save_restore_size) == 80);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, ctl_stack_size) == 84);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, sdma_engine_id) == 88);
    assert!(offset_of!(KfdIoctlCreateQueueArgs, pad) == 92);

    assert!(size_of::<KfdIoctlDestroyQueueArgs>() == 8);
    assert!(align_of::<KfdIoctlDestroyQueueArgs>() == 4);
    assert!(offset_of!(KfdIoctlDestroyQueueArgs, queue_id) == 0);
    assert!(offset_of!(KfdIoctlDestroyQueueArgs, pad) == 4);

    assert!(size_of::<KfdIoctlUpdateQueueArgs>() == 24);
    assert!(align_of::<KfdIoctlUpdateQueueArgs>() == 8);
    assert!(offset_of!(KfdIoctlUpdateQueueArgs, ring_base_address) == 0);
    assert!(offset_of!(KfdIoctlUpdateQueueArgs, queue_id) == 8);
    assert!(offset_of!(KfdIoctlUpdateQueueArgs, ring_size) == 12);
    assert!(offset_of!(KfdIoctlUpdateQueueArgs, queue_percentage) == 16);
    assert!(offset_of!(KfdIoctlUpdateQueueArgs, queue_priority) == 20);

    assert!(size_of::<KfdIoctlAcquireVmArgs>() == 8);
    assert!(align_of::<KfdIoctlAcquireVmArgs>() == 4);
    assert!(offset_of!(KfdIoctlAcquireVmArgs, drm_fd) == 0);
    assert!(offset_of!(KfdIoctlAcquireVmArgs, gpu_id) == 4);

    assert!(size_of::<KfdIoctlAllocMemoryOfGpuArgs>() == 40);
    assert!(align_of::<KfdIoctlAllocMemoryOfGpuArgs>() == 8);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, va_addr) == 0);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, size) == 8);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, handle) == 16);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, mmap_offset) == 24);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, gpu_id) == 32);
    assert!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, flags) == 36);

    assert!(size_of::<KfdIoctlFreeMemoryOfGpuArgs>() == 8);
    assert!(align_of::<KfdIoctlFreeMemoryOfGpuArgs>() == 8);
    assert!(offset_of!(KfdIoctlFreeMemoryOfGpuArgs, handle) == 0);

    assert!(size_of::<KfdIoctlMapMemoryToGpuArgs>() == 24);
    assert!(align_of::<KfdIoctlMapMemoryToGpuArgs>() == 8);
    assert!(offset_of!(KfdIoctlMapMemoryToGpuArgs, handle) == 0);
    assert!(offset_of!(KfdIoctlMapMemoryToGpuArgs, device_ids_array_ptr) == 8);
    assert!(offset_of!(KfdIoctlMapMemoryToGpuArgs, n_devices) == 16);
    assert!(offset_of!(KfdIoctlMapMemoryToGpuArgs, n_success) == 20);

    assert!(size_of::<KfdIoctlUnmapMemoryFromGpuArgs>() == 24);
    assert!(align_of::<KfdIoctlUnmapMemoryFromGpuArgs>() == 8);
    assert!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, handle) == 0);
    assert!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, device_ids_array_ptr) == 8);
    assert!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, n_devices) == 16);
    assert!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, n_success) == 20);

    assert!(size_of::<KfdProcessDeviceApertures>() == 56);
    assert!(align_of::<KfdProcessDeviceApertures>() == 8);
    assert!(offset_of!(KfdProcessDeviceApertures, lds_base) == 0);
    assert!(offset_of!(KfdProcessDeviceApertures, lds_limit) == 8);
    assert!(offset_of!(KfdProcessDeviceApertures, scratch_base) == 16);
    assert!(offset_of!(KfdProcessDeviceApertures, scratch_limit) == 24);
    assert!(offset_of!(KfdProcessDeviceApertures, gpuvm_base) == 32);
    assert!(offset_of!(KfdProcessDeviceApertures, gpuvm_limit) == 40);
    assert!(offset_of!(KfdProcessDeviceApertures, gpu_id) == 48);
    assert!(offset_of!(KfdProcessDeviceApertures, pad) == 52);

    assert!(size_of::<KfdIoctlGetProcessAperturesNewArgs>() == 16);
    assert!(align_of::<KfdIoctlGetProcessAperturesNewArgs>() == 8);
    assert!(
        offset_of!(
            KfdIoctlGetProcessAperturesNewArgs,
            kfd_process_device_apertures_ptr
        ) == 0
    );
    assert!(offset_of!(KfdIoctlGetProcessAperturesNewArgs, num_of_nodes) == 8);
    assert!(offset_of!(KfdIoctlGetProcessAperturesNewArgs, pad) == 12);

    assert!(size_of::<KfdIoctlSetXnackModeArgs>() == 4);
    assert!(align_of::<KfdIoctlSetXnackModeArgs>() == 4);
    assert!(offset_of!(KfdIoctlSetXnackModeArgs, xnack_enabled) == 0);

    assert!(size_of::<KfdIoctlSmiEventsArgs>() == 8);
    assert!(align_of::<KfdIoctlSmiEventsArgs>() == 4);
    assert!(offset_of!(KfdIoctlSmiEventsArgs, gpu_id) == 0);
    assert!(offset_of!(KfdIoctlSmiEventsArgs, anon_fd) == 4);

    assert!(AMDKFD_IOC_GET_VERSION == 0x8008_4b01);
    assert!(AMDKFD_IOC_CREATE_QUEUE == 0xc060_4b02);
    assert!(AMDKFD_IOC_DESTROY_QUEUE == 0xc008_4b03);
    assert!(AMDKFD_IOC_UPDATE_QUEUE == 0x4018_4b07);
    assert!(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW == 0xc010_4b14);
    assert!(AMDKFD_IOC_ACQUIRE_VM == 0x4008_4b15);
    assert!(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU == 0xc028_4b16);
    assert!(AMDKFD_IOC_FREE_MEMORY_OF_GPU == 0x4008_4b17);
    assert!(AMDKFD_IOC_MAP_MEMORY_TO_GPU == 0xc018_4b18);
    assert!(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU == 0xc018_4b19);
    assert!(AMDKFD_IOC_SET_XNACK_MODE == 0xc004_4b21);
    assert!(AMDKFD_IOC_SMI_EVENTS == 0xc008_4b1f);
    assert!(KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT == 0x8400_0002);
    assert!(KFD_ALLOC_MEMORY_FLAGS_KERNARG == 0x8600_0002);
    assert!(KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE == 0x8e00_0002);
    assert!(KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE == 0xc400_0002);
    assert!(KFD_SMI_EVENT_GPU_RESET_MASK == 0x0c);
};
