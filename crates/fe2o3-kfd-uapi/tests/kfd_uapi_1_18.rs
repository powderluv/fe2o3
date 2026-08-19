use core::mem::{align_of, offset_of, size_of};

use fe2o3_kfd_uapi::{
    AMDKFD_IOC_ACQUIRE_VM, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, AMDKFD_IOC_FREE_MEMORY_OF_GPU,
    AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, AMDKFD_IOC_GET_VERSION, AMDKFD_IOC_MAP_MEMORY_TO_GPU,
    AMDKFD_IOC_SET_XNACK_MODE, AMDKFD_IOC_SMI_EVENTS, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
    AMDKFD_IOCTL_BASE, IoctlDirection, KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE,
    KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE, KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT,
    KFD_ALLOC_MEMORY_FLAGS_KERNARG, KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM,
    KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
    KFD_IOC_ALLOC_MEM_FLAGS_GTT, KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE, KFD_IOCTL_MAJOR_VERSION,
    KFD_IOCTL_MAX_ADMITTED_MINOR_VERSION, KFD_IOCTL_MIN_ADMITTED_MINOR_VERSION,
    KFD_IOCTL_MINOR_VERSION, KFD_MEMORY_LIFECYCLE_SCHEMA_ID, KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST,
    KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256, KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES,
    KFD_SMI_EVENT_GPU_POST_RESET, KFD_SMI_EVENT_GPU_PRE_RESET, KFD_SMI_EVENT_GPU_RESET_MASK,
    KFD_SMI_EVENT_MSG_SIZE, KFD_UAPI_CHARDEV_SOURCE_SHA256, KFD_UAPI_DEVICE_SOURCE_SHA256,
    KFD_UAPI_GPUVM_SOURCE_SHA256, KFD_UAPI_SCHEMA_ID, KFD_UAPI_SCHEMA_MANIFEST,
    KFD_UAPI_SCHEMA_MANIFEST_SHA256, KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
    KFD_UAPI_SMI_EVENTS_SOURCE_SHA256, KFD_UAPI_SOURCE_HEADER_SHA256, KFD_XNACK_MODE_DISABLED,
    KFD_XNACK_MODE_ENABLED, KFD_XNACK_MODE_QUERY, KfdAllocMemoryFlags, KfdAllocMemoryFlagsError,
    KfdIoctlAcquireVmArgs, KfdIoctlAllocMemoryOfGpuArgs, KfdIoctlFreeMemoryOfGpuArgs,
    KfdIoctlGetProcessAperturesNewArgs, KfdIoctlGetVersionArgs, KfdIoctlMapMemoryToGpuArgs,
    KfdIoctlSetXnackModeArgs, KfdIoctlSmiEventsArgs, KfdIoctlUnmapMemoryFromGpuArgs,
    KfdProcessDeviceApertures, KfdUapiVersion, KfdUapiVersionError, admit_kfd_alloc_memory_flags,
    encode_ioctl, negotiate_kfd_uapi_version,
};
use sha2::{Digest, Sha256};

#[test]
fn schema_identity_is_linux_kfd_1_18() {
    assert_eq!(KFD_UAPI_SCHEMA_ID, "linux-kfd-uapi-1.18-generic-ioc-v1");
    assert_eq!(KFD_IOCTL_MAJOR_VERSION, 1);
    assert_eq!(KFD_IOCTL_MINOR_VERSION, 18);
    assert_eq!(KFD_IOCTL_MIN_ADMITTED_MINOR_VERSION, 18);
    assert_eq!(KFD_IOCTL_MAX_ADMITTED_MINOR_VERSION, 18);
    assert_eq!(
        KFD_UAPI_SOURCE_HEADER_SHA256,
        "b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d"
    );
    assert_eq!(
        KFD_UAPI_SMI_EVENTS_SOURCE_SHA256,
        "2d786562fe1e97b8257841b755106c8bce47658a2aa3b439ce4e0178323004bd"
    );
    assert_eq!(
        KFD_UAPI_DEVICE_SOURCE_SHA256,
        "ccf20227c5cdd5b258758f50f61bbc1008a09ea776c101f035f83963e7d23037"
    );
    assert_eq!(
        KFD_UAPI_CHARDEV_SOURCE_SHA256,
        "f9a8805c5d479faee25e457051aa428e4bb523ecf1c7b1618a6a5f79ca5d7bba"
    );
    let manifest_digest = Sha256::digest(KFD_UAPI_SCHEMA_MANIFEST);
    assert_eq!(&manifest_digest[..], &KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES);
    assert_eq!(
        KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
        [
            0xe4, 0xaa, 0xd5, 0xd8, 0xe3, 0x17, 0x7e, 0xa6, 0xd7, 0x02, 0x98, 0xad, 0xab, 0x77,
            0x41, 0xc3, 0x77, 0xcb, 0x09, 0x13, 0x73, 0x55, 0x3c, 0xe6, 0x89, 0xf3, 0x52, 0x5e,
            0x75, 0x14, 0xd9, 0xb4,
        ]
    );
    assert_eq!(
        KFD_UAPI_SCHEMA_MANIFEST_SHA256,
        "e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4"
    );
}

#[test]
fn memory_lifecycle_schema_composes_with_frozen_base_schema() {
    assert_eq!(
        KFD_MEMORY_LIFECYCLE_SCHEMA_ID,
        "linux-kfd-memory-lifecycle-1.18-generic-ioc-v1"
    );
    assert_eq!(
        KFD_UAPI_GPUVM_SOURCE_SHA256,
        "c7cca2ee47a08c99bb73906662d82dd7d0b5738468fbef54848e5e6dd62ba50d"
    );
    assert!(KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST.contains(&format!(
        "base_schema_manifest_sha256={KFD_UAPI_SCHEMA_MANIFEST_SHA256}\n"
    )));
    assert!(KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST.contains(&format!(
        "source_header_sha256={KFD_UAPI_SOURCE_HEADER_SHA256}\n"
    )));
    assert!(KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST.contains(&format!(
        "gpuvm_source_sha256={KFD_UAPI_GPUVM_SOURCE_SHA256}\n"
    )));

    let manifest_digest = Sha256::digest(KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST);
    assert_eq!(
        &manifest_digest[..],
        &KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES
    );
    assert_eq!(
        KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES,
        [
            0xe2, 0xd6, 0x98, 0x7b, 0x7c, 0x8e, 0x61, 0xa4, 0x05, 0xb2, 0xf7, 0x75, 0xd5, 0xd0,
            0x04, 0xf4, 0x58, 0xa0, 0x96, 0x24, 0x14, 0x59, 0xe4, 0xcf, 0xdf, 0x90, 0xbd, 0x44,
            0x97, 0xf4, 0xd5, 0x8a,
        ]
    );
    assert_eq!(
        KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256,
        "e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a"
    );
}

#[test]
fn get_version_layout_matches_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdIoctlGetVersionArgs>(), 8);
    assert_eq!(align_of::<KfdIoctlGetVersionArgs>(), 4);
    assert_eq!(offset_of!(KfdIoctlGetVersionArgs, major_version), 0);
    assert_eq!(offset_of!(KfdIoctlGetVersionArgs, minor_version), 4);
}

#[test]
fn acquire_vm_layout_matches_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdIoctlAcquireVmArgs>(), 8);
    assert_eq!(align_of::<KfdIoctlAcquireVmArgs>(), 4);
    assert_eq!(offset_of!(KfdIoctlAcquireVmArgs, drm_fd), 0);
    assert_eq!(offset_of!(KfdIoctlAcquireVmArgs, gpu_id), 4);
}

#[test]
fn memory_lifecycle_layouts_match_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdIoctlAllocMemoryOfGpuArgs>(), 40);
    assert_eq!(align_of::<KfdIoctlAllocMemoryOfGpuArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, va_addr), 0);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, size), 8);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, handle), 16);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, mmap_offset), 24);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, gpu_id), 32);
    assert_eq!(offset_of!(KfdIoctlAllocMemoryOfGpuArgs, flags), 36);

    assert_eq!(size_of::<KfdIoctlFreeMemoryOfGpuArgs>(), 8);
    assert_eq!(align_of::<KfdIoctlFreeMemoryOfGpuArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlFreeMemoryOfGpuArgs, handle), 0);

    assert_eq!(size_of::<KfdIoctlMapMemoryToGpuArgs>(), 24);
    assert_eq!(align_of::<KfdIoctlMapMemoryToGpuArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlMapMemoryToGpuArgs, handle), 0);
    assert_eq!(
        offset_of!(KfdIoctlMapMemoryToGpuArgs, device_ids_array_ptr),
        8
    );
    assert_eq!(offset_of!(KfdIoctlMapMemoryToGpuArgs, n_devices), 16);
    assert_eq!(offset_of!(KfdIoctlMapMemoryToGpuArgs, n_success), 20);

    assert_eq!(size_of::<KfdIoctlUnmapMemoryFromGpuArgs>(), 24);
    assert_eq!(align_of::<KfdIoctlUnmapMemoryFromGpuArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, handle), 0);
    assert_eq!(
        offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, device_ids_array_ptr),
        8
    );
    assert_eq!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, n_devices), 16);
    assert_eq!(offset_of!(KfdIoctlUnmapMemoryFromGpuArgs, n_success), 20);
}

#[test]
fn admitted_memory_flags_match_kfd_uapi_1_18_golden() {
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0x0000_0002);
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE, 0x8000_0000);
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, 0x4000_0000);
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM, 0x0800_0000);
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0x0400_0000);
    assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED, 0x0200_0000);

    assert_eq!(KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT, 0x8400_0002);
    assert_eq!(KFD_ALLOC_MEMORY_FLAGS_KERNARG, 0x8600_0002);
    assert_eq!(KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE, 0x8e00_0002);
    assert_eq!(KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE, 0xc400_0002);

    assert_eq!(
        admit_kfd_alloc_memory_flags(KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT),
        Ok(KfdAllocMemoryFlags::HOST_VISIBLE_COHERENT)
    );
    assert_eq!(
        admit_kfd_alloc_memory_flags(KFD_ALLOC_MEMORY_FLAGS_KERNARG),
        Ok(KfdAllocMemoryFlags::KERNARG)
    );
    assert_eq!(
        admit_kfd_alloc_memory_flags(KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE),
        Ok(KfdAllocMemoryFlags::AQL_QUEUE)
    );
    assert_eq!(
        admit_kfd_alloc_memory_flags(KFD_ALLOC_MEMORY_FLAGS_EXECUTABLE),
        Ok(KfdAllocMemoryFlags::EXECUTABLE)
    );
}

#[test]
fn memory_flag_admission_rejects_hostile_and_unreviewed_patterns() {
    for flags in [
        0,
        1,      // VRAM
        1 << 2, // USERPTR/SVM
        KFD_IOC_ALLOC_MEM_FLAGS_GTT,
        KFD_ALLOC_MEMORY_FLAGS_HOST_VISIBLE_COHERENT | 1,
        KFD_ALLOC_MEMORY_FLAGS_KERNARG | (1 << 24),
        KFD_ALLOC_MEMORY_FLAGS_AQL_QUEUE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
        u32::MAX,
    ] {
        assert_eq!(
            admit_kfd_alloc_memory_flags(flags),
            Err(KfdAllocMemoryFlagsError::Unsupported { flags })
        );
    }
}

#[test]
fn process_apertures_layouts_match_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdProcessDeviceApertures>(), 56);
    assert_eq!(align_of::<KfdProcessDeviceApertures>(), 8);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, lds_base), 0);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, lds_limit), 8);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, scratch_base), 16);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, scratch_limit), 24);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, gpuvm_base), 32);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, gpuvm_limit), 40);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, gpu_id), 48);
    assert_eq!(offset_of!(KfdProcessDeviceApertures, pad), 52);

    assert_eq!(size_of::<KfdIoctlGetProcessAperturesNewArgs>(), 16);
    assert_eq!(align_of::<KfdIoctlGetProcessAperturesNewArgs>(), 8);
    assert_eq!(
        offset_of!(
            KfdIoctlGetProcessAperturesNewArgs,
            kfd_process_device_apertures_ptr
        ),
        0
    );
    assert_eq!(
        offset_of!(KfdIoctlGetProcessAperturesNewArgs, num_of_nodes),
        8
    );
    assert_eq!(offset_of!(KfdIoctlGetProcessAperturesNewArgs, pad), 12);
}

#[test]
fn set_xnack_mode_layout_matches_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdIoctlSetXnackModeArgs>(), 4);
    assert_eq!(align_of::<KfdIoctlSetXnackModeArgs>(), 4);
    assert_eq!(offset_of!(KfdIoctlSetXnackModeArgs, xnack_enabled), 0);
}

#[test]
fn smi_events_layout_and_reset_mask_match_kfd_uapi_1_18_golden() {
    assert_eq!(size_of::<KfdIoctlSmiEventsArgs>(), 8);
    assert_eq!(align_of::<KfdIoctlSmiEventsArgs>(), 4);
    assert_eq!(offset_of!(KfdIoctlSmiEventsArgs, gpu_id), 0);
    assert_eq!(offset_of!(KfdIoctlSmiEventsArgs, anon_fd), 4);
    assert_eq!(KFD_SMI_EVENT_GPU_PRE_RESET, 3);
    assert_eq!(KFD_SMI_EVENT_GPU_POST_RESET, 4);
    assert_eq!(KFD_SMI_EVENT_GPU_RESET_MASK, 0x0c);
    assert_eq!(KFD_SMI_EVENT_MSG_SIZE, 96);
}

#[test]
fn ioctl_numbers_match_linux_generic_ioc_golden() {
    assert_eq!(AMDKFD_IOC_GET_VERSION, 0x8008_4b01);
    assert_eq!(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, 0xc010_4b14);
    assert_eq!(AMDKFD_IOC_ACQUIRE_VM, 0x4008_4b15);
    assert_eq!(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, 0xc028_4b16);
    assert_eq!(AMDKFD_IOC_FREE_MEMORY_OF_GPU, 0x4008_4b17);
    assert_eq!(AMDKFD_IOC_MAP_MEMORY_TO_GPU, 0xc018_4b18);
    assert_eq!(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, 0xc018_4b19);
    assert_eq!(AMDKFD_IOC_SET_XNACK_MODE, 0xc004_4b21);
    assert_eq!(AMDKFD_IOC_SMI_EVENTS, 0xc008_4b1f);

    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x14,
            size_of::<KfdIoctlGetProcessAperturesNewArgs>(),
        ),
        Some(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::Read,
            AMDKFD_IOCTL_BASE,
            0x01,
            size_of::<KfdIoctlGetVersionArgs>(),
        ),
        Some(AMDKFD_IOC_GET_VERSION),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::Write,
            AMDKFD_IOCTL_BASE,
            0x15,
            size_of::<KfdIoctlAcquireVmArgs>(),
        ),
        Some(AMDKFD_IOC_ACQUIRE_VM),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x16,
            size_of::<KfdIoctlAllocMemoryOfGpuArgs>(),
        ),
        Some(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::Write,
            AMDKFD_IOCTL_BASE,
            0x17,
            size_of::<KfdIoctlFreeMemoryOfGpuArgs>(),
        ),
        Some(AMDKFD_IOC_FREE_MEMORY_OF_GPU),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x18,
            size_of::<KfdIoctlMapMemoryToGpuArgs>(),
        ),
        Some(AMDKFD_IOC_MAP_MEMORY_TO_GPU),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x19,
            size_of::<KfdIoctlUnmapMemoryFromGpuArgs>(),
        ),
        Some(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x21,
            size_of::<KfdIoctlSetXnackModeArgs>(),
        ),
        Some(AMDKFD_IOC_SET_XNACK_MODE),
    );
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0x1f,
            size_of::<KfdIoctlSmiEventsArgs>(),
        ),
        Some(AMDKFD_IOC_SMI_EVENTS),
    );
}

#[test]
fn ioctl_encoder_rejects_unrepresentable_payload_size() {
    let first_unrepresentable_size = 1 << 14;
    assert_eq!(
        encode_ioctl(
            IoctlDirection::ReadWrite,
            AMDKFD_IOCTL_BASE,
            0xff,
            first_unrepresentable_size,
        ),
        None,
    );
}

#[test]
fn exact_reviewed_version_produces_admission_evidence() {
    let admitted = negotiate_kfd_uapi_version(KfdUapiVersion::new(1, 18)).unwrap();
    assert_eq!(admitted.reported_version(), KfdUapiVersion::new(1, 18));
    assert_eq!(admitted.schema_id(), KFD_UAPI_SCHEMA_ID);
    assert_eq!(
        admitted.schema_manifest_sha256(),
        KFD_UAPI_SCHEMA_MANIFEST_SHA256
    );
    assert_eq!(admitted.acquire_vm_request(), AMDKFD_IOC_ACQUIRE_VM);
    assert_eq!(
        admitted.get_process_apertures_new_request(),
        AMDKFD_IOC_GET_PROCESS_APERTURES_NEW
    );
    assert_eq!(admitted.set_xnack_mode_request(), AMDKFD_IOC_SET_XNACK_MODE);
    assert_eq!(admitted.smi_events_request(), AMDKFD_IOC_SMI_EVENTS);
}

#[test]
fn version_negotiation_fails_closed() {
    assert_eq!(
        negotiate_kfd_uapi_version(KfdUapiVersion::new(0, 18)),
        Err(KfdUapiVersionError::UnsupportedMajor {
            reported: 0,
            admitted: 1,
        }),
    );
    assert_eq!(
        negotiate_kfd_uapi_version(KfdUapiVersion::new(1, 17)),
        Err(KfdUapiVersionError::MinorTooOld {
            reported: 17,
            minimum: 18,
        }),
    );
    assert_eq!(
        negotiate_kfd_uapi_version(KfdUapiVersion::new(1, 19)),
        Err(KfdUapiVersionError::MinorNewerThanReviewed {
            reported: 19,
            maximum: 18,
        }),
    );
    assert_eq!(
        negotiate_kfd_uapi_version(KfdUapiVersion::new(2, 0)),
        Err(KfdUapiVersionError::UnsupportedMajor {
            reported: 2,
            admitted: 1,
        }),
    );
}

#[test]
fn raw_argument_constructors_preserve_uapi_values() {
    let version = KfdIoctlGetVersionArgs {
        major_version: 1,
        minor_version: 18,
    };
    assert_eq!(version.reported_version(), KfdUapiVersion::new(1, 18));

    let acquire_vm = KfdIoctlAcquireVmArgs::new(27, 9_812);
    assert_eq!(acquire_vm.drm_fd, 27);
    assert_eq!(acquire_vm.gpu_id, 9_812);

    let alloc = KfdIoctlAllocMemoryOfGpuArgs::new(
        0x7f00_0000_0000,
        0x20_0000,
        9_812,
        KfdAllocMemoryFlags::KERNARG,
    );
    assert_eq!(alloc.va_addr, 0x7f00_0000_0000);
    assert_eq!(alloc.size, 0x20_0000);
    assert_eq!(alloc.handle, 0);
    assert_eq!(alloc.mmap_offset, 0);
    assert_eq!(alloc.gpu_id, 9_812);
    assert_eq!(alloc.flags, KFD_ALLOC_MEMORY_FLAGS_KERNARG);

    let free = KfdIoctlFreeMemoryOfGpuArgs::new(0xfedc_ba98_7654_3210);
    assert_eq!(free.handle, 0xfedc_ba98_7654_3210);

    let map = KfdIoctlMapMemoryToGpuArgs::initial(0x1234_5678_9abc_def0, 0x7fff_ffff_f000, 8);
    assert_eq!(map.handle, 0x1234_5678_9abc_def0);
    assert_eq!(map.device_ids_array_ptr, 0x7fff_ffff_f000);
    assert_eq!(map.n_devices, 8);
    assert_eq!(map.n_success, 0);

    let map_retry = KfdIoctlMapMemoryToGpuArgs::retry(u64::MAX, u64::MAX - 1, 8, 5);
    assert_eq!(map_retry.handle, u64::MAX);
    assert_eq!(map_retry.device_ids_array_ptr, u64::MAX - 1);
    assert_eq!(map_retry.n_devices, 8);
    assert_eq!(map_retry.n_success, 5);

    let unmap = KfdIoctlUnmapMemoryFromGpuArgs::initial(0x1234_5678_9abc_def0, 0x7fff_ffff_f000, 8);
    assert_eq!(unmap.n_success, 0);

    let unmap_retry = KfdIoctlUnmapMemoryFromGpuArgs::retry(u64::MAX, 1, u32::MAX, u32::MAX);
    assert_eq!(unmap_retry.handle, u64::MAX);
    assert_eq!(unmap_retry.device_ids_array_ptr, 1);
    assert_eq!(unmap_retry.n_devices, u32::MAX);
    assert_eq!(unmap_retry.n_success, u32::MAX);

    let apertures = KfdIoctlGetProcessAperturesNewArgs::new(0x1234_5000, 16);
    assert_eq!(apertures.kfd_process_device_apertures_ptr, 0x1234_5000);
    assert_eq!(apertures.num_of_nodes, 16);
    assert_eq!(apertures.pad, 0);

    let smi = KfdIoctlSmiEventsArgs::new(9_812);
    assert_eq!(smi.gpu_id, 9_812);
    assert_eq!(smi.anon_fd, u32::MAX);

    assert_eq!(
        KfdIoctlSetXnackModeArgs::query().xnack_enabled,
        KFD_XNACK_MODE_QUERY
    );
    assert_eq!(
        KfdIoctlSetXnackModeArgs::set(false).xnack_enabled,
        KFD_XNACK_MODE_DISABLED
    );
    assert_eq!(
        KfdIoctlSetXnackModeArgs::set(true).xnack_enabled,
        KFD_XNACK_MODE_ENABLED
    );
}
