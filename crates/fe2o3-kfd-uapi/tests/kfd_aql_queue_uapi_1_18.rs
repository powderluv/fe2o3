use core::mem::{align_of, offset_of, size_of};

use fe2o3_kfd_uapi::{
    AMDKFD_IOC_CREATE_QUEUE, AMDKFD_IOC_DESTROY_QUEUE, AMDKFD_IOC_UPDATE_QUEUE,
    KFD_AQL_QUEUE_AMDGPU_GFX9_HEADER_SHA256, KFD_AQL_QUEUE_AMDGPU_GFX9_SOURCE_SHA256,
    KFD_AQL_QUEUE_BUFFER_SOURCE_SHA256, KFD_AQL_QUEUE_DQM_GFX9_SOURCE_SHA256,
    KFD_AQL_QUEUE_DQM_HEADER_SHA256, KFD_AQL_QUEUE_DQM_SOURCE_SHA256,
    KFD_AQL_QUEUE_GC9_OFFSET_HEADER_SHA256, KFD_AQL_QUEUE_GC9_SH_MASK_HEADER_SHA256,
    KFD_AQL_QUEUE_GC943_SH_MASK_HEADER_SHA256, KFD_AQL_QUEUE_GFX9_MQD_SOURCE_SHA256,
    KFD_AQL_QUEUE_KERNEL_QUEUE_SOURCE_SHA256, KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_ID,
    KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST, KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256,
    KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES, KFD_AQL_QUEUE_MQD_MANAGER_HEADER_SHA256,
    KFD_AQL_QUEUE_PACKET_MANAGER_SOURCE_SHA256, KFD_AQL_QUEUE_PQM_SOURCE_SHA256,
    KFD_AQL_QUEUE_PRIV_SOURCE_SHA256, KFD_AQL_QUEUE_V9_STRUCTS_HEADER_SHA256,
    KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, KFD_MAX_QUEUE_PERCENTAGE, KFD_MAX_QUEUE_PRIORITY,
    KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256, KFD_MIN_QUEUE_RING_SIZE,
    KFD_UAPI_SCHEMA_MANIFEST_SHA256, KfdAqlComputeQueueBuffers, KfdAqlQueueRingAddressError,
    KfdAqlQueueRingSizeError, KfdIoctlCreateQueueArgs, KfdIoctlDestroyQueueArgs,
    KfdIoctlUpdateQueueArgs, KfdQueuePercentageError, KfdQueuePriorityError,
    admit_kfd_aql_queue_ring_address, admit_kfd_aql_queue_ring_size, admit_kfd_queue_percentage,
    admit_kfd_queue_priority,
};
use sha2::{Digest, Sha256};

#[test]
fn queue_schema_is_separate_and_composes_with_frozen_prerequisites() {
    assert_eq!(
        KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_ID,
        "linux-kfd-aql-queue-lifecycle-1.18-generic-ioc-v1"
    );
    assert_eq!(
        KFD_UAPI_SCHEMA_MANIFEST_SHA256,
        "e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4"
    );
    assert_eq!(
        KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST_SHA256,
        "e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a"
    );
    assert!(KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST.contains(
        "base_schema_manifest_sha256=e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4"
    ));
    assert!(KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST.contains(
        "memory_schema_manifest_sha256=e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a"
    ));

    let digest = Sha256::digest(KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST);
    assert_eq!(hex(&digest), KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256);
    assert_eq!(
        &digest[..],
        &KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST_SHA256_BYTES
    );
}

#[test]
fn queue_semantic_sources_are_exactly_pinned() {
    assert_eq!(
        KFD_AQL_QUEUE_BUFFER_SOURCE_SHA256,
        "fb4b2a5c9e6981222873bcd7aca7e9c1397cba8f1a6b33634d2a48d4427fe062"
    );
    assert_eq!(
        KFD_AQL_QUEUE_PQM_SOURCE_SHA256,
        "8526e258824dbe145e4209cf0fed26463729234ba24369f39e3413e7e6e028db"
    );
    assert_eq!(
        KFD_AQL_QUEUE_DQM_SOURCE_SHA256,
        "d61e53a78c1855c4badefbebb6c6ec52702be8cfe072253341c277337641c682"
    );
    assert_eq!(
        KFD_AQL_QUEUE_GFX9_MQD_SOURCE_SHA256,
        "21166e9dbe2a4c24cbcd6f9ff6193aa093230e91fbafc8b4ac4eee1465cd2c9e"
    );
    assert_eq!(
        KFD_AQL_QUEUE_PRIV_SOURCE_SHA256,
        "f991330031c14725b2be0636ec1896ab530dc3d07d530ebd4f47efff97a82a99"
    );
    assert_eq!(
        KFD_AQL_QUEUE_DQM_GFX9_SOURCE_SHA256,
        "53021a6f8211212f872545403e200d34d2e8c49b1cbdd17e382ae7baa43e52f2"
    );
    assert_eq!(
        KFD_AQL_QUEUE_PACKET_MANAGER_SOURCE_SHA256,
        "1ed642990cbb7d4cdbde211fee571318e233c19744ea1663d8eb68946c1310dd"
    );
    assert_eq!(
        KFD_AQL_QUEUE_KERNEL_QUEUE_SOURCE_SHA256,
        "13e5d3634bcfed2ae871d8da0700cde47d8671eb014831b5d1ca95ed5a22fb36"
    );
    assert_eq!(
        KFD_AQL_QUEUE_MQD_MANAGER_HEADER_SHA256,
        "61ea7d4a13fb3168d0f026ecb13b13cf5846c86f233289043728b62ac9068605"
    );
    assert_eq!(
        KFD_AQL_QUEUE_DQM_HEADER_SHA256,
        "9e43b8f41ad89d1dd21fddf38dff4182f09b01218778f8278a743eacb72ceadd"
    );
    assert_eq!(
        KFD_AQL_QUEUE_V9_STRUCTS_HEADER_SHA256,
        "18f8e59e4cab35d579d2e3f9fc4eadffd81d518d586065de4d9d0ab4fcc131d7"
    );
    assert_eq!(
        KFD_AQL_QUEUE_AMDGPU_GFX9_SOURCE_SHA256,
        "d112169b3231439086da4943c7675bb4aeddb111b483a687fdd95794710ab27c"
    );
    assert_eq!(
        KFD_AQL_QUEUE_AMDGPU_GFX9_HEADER_SHA256,
        "97bc6cd046c9c2495962d26d455e5231d95b0503385354177c366ea21fa9ed2e"
    );
    assert_eq!(
        KFD_AQL_QUEUE_GC9_OFFSET_HEADER_SHA256,
        "dde287260e0b63eecfd7b723c1fdfaf9a3da7155f0ccd331385b9acc09433aa5"
    );
    assert_eq!(
        KFD_AQL_QUEUE_GC9_SH_MASK_HEADER_SHA256,
        "f67f3f753231a53e82e39783313605cd382eb9727f2cda775d6e849a7c38063e"
    );
    assert_eq!(
        KFD_AQL_QUEUE_GC943_SH_MASK_HEADER_SHA256,
        "8ee3fb2c721703a1643c118502e2900bd622b4d8d287103bd53922f92d35611b"
    );

    for pinned_pair in [
        "gfx9_dqm_source=amd/amdkfd/kfd_device_queue_manager_v9.c\ngfx9_dqm_source_sha256=53021a6f8211212f872545403e200d34d2e8c49b1cbdd17e382ae7baa43e52f2",
        "packet_manager_source=amd/amdkfd/kfd_packet_manager.c\npacket_manager_source_sha256=1ed642990cbb7d4cdbde211fee571318e233c19744ea1663d8eb68946c1310dd",
        "kernel_queue_source=amd/amdkfd/kfd_kernel_queue.c\nkernel_queue_source_sha256=13e5d3634bcfed2ae871d8da0700cde47d8671eb014831b5d1ca95ed5a22fb36",
        "mqd_manager_header=amd/amdkfd/kfd_mqd_manager.h\nmqd_manager_header_sha256=61ea7d4a13fb3168d0f026ecb13b13cf5846c86f233289043728b62ac9068605",
        "dqm_header=amd/amdkfd/kfd_device_queue_manager.h\ndqm_header_sha256=9e43b8f41ad89d1dd21fddf38dff4182f09b01218778f8278a743eacb72ceadd",
        "v9_structs_header=amd/include/v9_structs.h\nv9_structs_header_sha256=18f8e59e4cab35d579d2e3f9fc4eadffd81d518d586065de4d9d0ab4fcc131d7",
        "amdgpu_gfx9_source=amd/amdgpu/amdgpu_amdkfd_gfx_v9.c\namdgpu_gfx9_source_sha256=d112169b3231439086da4943c7675bb4aeddb111b483a687fdd95794710ab27c",
        "amdgpu_gfx9_header=amd/amdgpu/amdgpu_amdkfd_gfx_v9.h\namdgpu_gfx9_header_sha256=97bc6cd046c9c2495962d26d455e5231d95b0503385354177c366ea21fa9ed2e",
        "gc9_offset_header=amd/include/asic_reg/gc/gc_9_0_offset.h\ngc9_offset_header_sha256=dde287260e0b63eecfd7b723c1fdfaf9a3da7155f0ccd331385b9acc09433aa5",
        "gc9_sh_mask_header=amd/include/asic_reg/gc/gc_9_0_sh_mask.h\ngc9_sh_mask_header_sha256=f67f3f753231a53e82e39783313605cd382eb9727f2cda775d6e849a7c38063e",
        "gc943_sh_mask_header=amd/include/asic_reg/gc/gc_9_4_3_sh_mask.h\ngc943_sh_mask_header_sha256=8ee3fb2c721703a1643c118502e2900bd622b4d8d287103bd53922f92d35611b",
    ] {
        assert!(KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST.contains(pinned_pair));
    }
}

#[test]
fn queue_layouts_match_active_header_oracle() {
    assert_eq!(size_of::<KfdIoctlCreateQueueArgs>(), 96);
    assert_eq!(align_of::<KfdIoctlCreateQueueArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, ring_base_address), 0);
    assert_eq!(
        offset_of!(KfdIoctlCreateQueueArgs, write_pointer_address),
        8
    );
    assert_eq!(
        offset_of!(KfdIoctlCreateQueueArgs, read_pointer_address),
        16
    );
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, doorbell_offset), 24);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, ring_size), 32);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, gpu_id), 36);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, queue_type), 40);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, queue_percentage), 44);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, queue_priority), 48);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, queue_id), 52);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, eop_buffer_address), 56);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, eop_buffer_size), 64);
    assert_eq!(
        offset_of!(KfdIoctlCreateQueueArgs, ctx_save_restore_address),
        72
    );
    assert_eq!(
        offset_of!(KfdIoctlCreateQueueArgs, ctx_save_restore_size),
        80
    );
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, ctl_stack_size), 84);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, sdma_engine_id), 88);
    assert_eq!(offset_of!(KfdIoctlCreateQueueArgs, pad), 92);

    assert_eq!(size_of::<KfdIoctlDestroyQueueArgs>(), 8);
    assert_eq!(align_of::<KfdIoctlDestroyQueueArgs>(), 4);
    assert_eq!(offset_of!(KfdIoctlDestroyQueueArgs, queue_id), 0);
    assert_eq!(offset_of!(KfdIoctlDestroyQueueArgs, pad), 4);

    assert_eq!(size_of::<KfdIoctlUpdateQueueArgs>(), 24);
    assert_eq!(align_of::<KfdIoctlUpdateQueueArgs>(), 8);
    assert_eq!(offset_of!(KfdIoctlUpdateQueueArgs, ring_base_address), 0);
    assert_eq!(offset_of!(KfdIoctlUpdateQueueArgs, queue_id), 8);
    assert_eq!(offset_of!(KfdIoctlUpdateQueueArgs, ring_size), 12);
    assert_eq!(offset_of!(KfdIoctlUpdateQueueArgs, queue_percentage), 16);
    assert_eq!(offset_of!(KfdIoctlUpdateQueueArgs, queue_priority), 20);
}

#[test]
fn queue_constants_and_requests_match_active_header_oracle() {
    assert_eq!(KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, 0x2);
    assert_eq!(KFD_MAX_QUEUE_PERCENTAGE, 100);
    assert_eq!(KFD_MAX_QUEUE_PRIORITY, 15);
    assert_eq!(KFD_MIN_QUEUE_RING_SIZE, 1024);
    assert_eq!(AMDKFD_IOC_CREATE_QUEUE, 0xc060_4b02);
    assert_eq!(AMDKFD_IOC_DESTROY_QUEUE, 0xc008_4b03);
    assert_eq!(AMDKFD_IOC_UPDATE_QUEUE, 0x4018_4b07);
}

#[test]
fn compute_aql_builder_fixes_type_outputs_and_reserved_fields() {
    let buffers = KfdAqlComputeQueueBuffers {
        ring_base_address: 0x1_0000,
        write_pointer_address: 0x2_0000,
        read_pointer_address: 0x2_1000,
        eop_buffer_address: 0x3_0000,
        eop_buffer_size: 0x1000,
        ctx_save_restore_address: 0x4_0000,
        ctx_save_restore_size: 0x8000,
        ctl_stack_size: 0x1000,
    };
    let args = KfdIoctlCreateQueueArgs::new_compute_aql(
        buffers,
        admit_kfd_aql_queue_ring_size(4096).unwrap(),
        7,
        admit_kfd_queue_percentage(100).unwrap(),
        admit_kfd_queue_priority(15).unwrap(),
    );

    assert_eq!(args.ring_base_address, buffers.ring_base_address);
    assert_eq!(args.write_pointer_address, buffers.write_pointer_address);
    assert_eq!(args.read_pointer_address, buffers.read_pointer_address);
    assert_eq!(args.ring_size, 4096);
    assert_eq!(args.gpu_id, 7);
    assert_eq!(args.queue_type, KFD_IOC_QUEUE_TYPE_COMPUTE_AQL);
    assert_eq!(args.queue_percentage, 100);
    assert_eq!(args.queue_priority, 15);
    assert_eq!(args.doorbell_offset, u64::MAX);
    assert_eq!(args.queue_id, u32::MAX);
    assert_eq!(args.sdma_engine_id, 0);
    assert_eq!(args.pad, 0);
}

#[test]
fn ring_size_admission_is_fail_closed() {
    for size in [KFD_MIN_QUEUE_RING_SIZE, 2048, 4096, 1 << 31] {
        assert_eq!(admit_kfd_aql_queue_ring_size(size).unwrap().bytes(), size);
    }
    for size in [0, 1, 512, 1023] {
        assert_eq!(
            admit_kfd_aql_queue_ring_size(size),
            Err(KfdAqlQueueRingSizeError::BelowMinimum {
                size,
                minimum: KFD_MIN_QUEUE_RING_SIZE,
            })
        );
    }
    for size in [1025, 1536, 4095, u32::MAX] {
        assert_eq!(
            admit_kfd_aql_queue_ring_size(size),
            Err(KfdAqlQueueRingSizeError::NotPowerOfTwo { size })
        );
    }
}

#[test]
fn percentage_and_priority_admission_rejects_extensions() {
    for percentage in [0, 1, 50, 100] {
        assert_eq!(
            admit_kfd_queue_percentage(percentage).unwrap().value(),
            percentage
        );
    }
    for percentage in [101, 0x100, 0xff00, u32::MAX] {
        assert_eq!(
            admit_kfd_queue_percentage(percentage),
            Err(KfdQueuePercentageError {
                percentage,
                maximum: KFD_MAX_QUEUE_PERCENTAGE,
            })
        );
    }
    for priority in [0, 1, 7, 15] {
        assert_eq!(
            admit_kfd_queue_priority(priority).unwrap().value(),
            priority
        );
    }
    for priority in [16, 0x100, u32::MAX] {
        assert_eq!(
            admit_kfd_queue_priority(priority),
            Err(KfdQueuePriorityError {
                priority,
                maximum: KFD_MAX_QUEUE_PRIORITY,
            })
        );
    }
}

#[test]
fn update_and_destroy_builders_preserve_only_reviewed_fields() {
    let update = KfdIoctlUpdateQueueArgs::reconfigure_compute_aql(
        19,
        admit_kfd_aql_queue_ring_address(0x8_0000).unwrap(),
        admit_kfd_aql_queue_ring_size(8192).unwrap(),
        admit_kfd_queue_percentage(0).unwrap(),
        admit_kfd_queue_priority(3).unwrap(),
    );
    assert_eq!(update.queue_id, 19);
    assert_eq!(update.ring_base_address, 0x8_0000);
    assert_eq!(update.ring_size, 8192);
    assert_eq!(update.queue_percentage, 0);
    assert_eq!(update.queue_priority, 3);

    let destroy = KfdIoctlDestroyQueueArgs::new(19);
    assert_eq!(destroy.queue_id, 19);
    assert_eq!(destroy.pad, 0);
}

#[test]
fn update_ring_address_and_disable_profiles_are_unambiguous() {
    assert_eq!(
        admit_kfd_aql_queue_ring_address(0),
        Err(KfdAqlQueueRingAddressError)
    );
    for address in [1, 0x1000, u64::MAX] {
        assert_eq!(
            admit_kfd_aql_queue_ring_address(address).unwrap().value(),
            address
        );
    }

    let disable = KfdIoctlUpdateQueueArgs::disable_compute_aql(
        19,
        admit_kfd_aql_queue_ring_size(8192).unwrap(),
        admit_kfd_queue_priority(3).unwrap(),
    );
    assert_eq!(disable.ring_base_address, 0);
    assert_eq!(disable.queue_id, 19);
    assert_eq!(disable.ring_size, 8192);
    assert_eq!(disable.queue_percentage, 0);
    assert_eq!(disable.queue_priority, 3);
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use core::fmt::Write;
        write!(&mut output, "{byte:02x}").unwrap();
    }
    output
}
