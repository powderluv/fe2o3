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
