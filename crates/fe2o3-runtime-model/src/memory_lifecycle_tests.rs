use alloc::{vec, vec::Vec};

use super::*;

const TEST_KFD_DYNAMIC_MAJOR: u32 = 511;

fn digest(seed: u8) -> IdentityDigestV1 {
    IdentityDigestV1::from_untrusted_bytes([seed; IDENTITY_DIGEST_BYTES_V1])
}

fn domain(seed: u8) -> DeviceObservationDomainIdV1 {
    DeviceObservationDomainIdV1::from_untrusted_digest(digest(seed))
}

fn profile() -> DeviceAdmissionProfileV1 {
    DeviceAdmissionProfileV1::gfx942_xnack_minus_spx_nps1_kfd_1_18_drm_3_64_0(
        DeviceAdmissionProfileIdV1::from_untrusted_digest(digest(2)),
        digest(3),
        digest(4),
    )
}

fn correlation(seed: u8) -> ModelCorrelatedDeviceV1 {
    let domain_id = domain(1);
    let epoch = ObservationEpochV1(9);
    let pci = PciAddressV1 {
        domain: 0,
        bus: seed,
        device: 1,
        function: 0,
    };
    UntrustedDeviceInventoryV1::from_untrusted_observations(
        UntrustedKfdObservationV1 {
            domain_id,
            epoch,
            node: DeviceNodeV1 {
                major: TEST_KFD_DYNAMIC_MAJOR,
                minor: KFD_DEVICE_MINOR_V1,
            },
            uapi_major: KFD_UAPI_MAJOR_V1,
            uapi_minor: KFD_UAPI_MINOR_V1,
            schema_identity: digest(3),
            xnack: XnackObservationV1::Disabled,
        },
        vec![UntrustedTopologyObservationV1 {
            domain_id,
            epoch,
            topology_node_id: u32::from(seed),
            kfd_gpu_id: u32::from(seed) + 1,
            gpu_unique_id: u64::from(seed) + 100,
            drm_render_minor: DRM_RENDER_MIN_MINOR_V1 + u32::from(seed),
            pci,
            vendor_id: AMD_PCI_VENDOR_ID_V1,
            device_id: MI300X_PCI_DEVICE_ID_V1,
            target: GpuTargetObservationV1::Gfx942,
            compute_partition: ComputePartitionObservationV1::Spx,
            memory_partition: MemoryPartitionObservationV1::Nps1,
        }],
        vec![UntrustedRenderObservationV1 {
            domain_id,
            epoch,
            node: DeviceNodeV1 {
                major: DRM_DEVICE_MAJOR_V1,
                minor: DRM_RENDER_MIN_MINOR_V1 + u32::from(seed),
            },
            gpu_unique_id: u64::from(seed) + 100,
            pci,
            vendor_id: AMD_PCI_VENDOR_ID_V1,
            device_id: MI300X_PCI_DEVICE_ID_V1,
            pci_revision_id: 0,
            drm_schema_identity: digest(4),
            driver_name: DrmDriverNameObservationV1::Amdgpu,
            drm_major: DRM_DRIVER_MAJOR_V1,
            drm_minor: DRM_DRIVER_MINOR_V1,
            drm_patch: DRM_DRIVER_PATCH_V1,
            acceleration_working: true,
            family: DrmFamilyObservationV1::AmdgpuFamilyAi,
        }],
    )
    .unwrap()
    .correlate_model_only(&profile())
    .unwrap()
}

fn vm_observation(device: ModelDeviceAdmissionV1, vm_id: u64) -> UntrustedVmObservationV1 {
    let correlated = device.correlation();
    UntrustedVmObservationV1 {
        domain_id: correlated.domain_id(),
        device: device.model_key(),
        vm_id: VmIdV1(vm_id),
        kfd_gpu_id: correlated.kfd_gpu_id(),
        render_node: correlated.render_node(),
        pci: correlated.identity().pci,
    }
}

#[derive(Clone, Copy)]
struct AdmissionFixture {
    first: ModelDeviceAdmissionV1,
    second: ModelDeviceAdmissionV1,
    first_vm: ModelVmAdmissionV1,
    second_vm: ModelVmAdmissionV1,
}

fn admissions() -> AdmissionFixture {
    let identity = DeviceIdentityStateV1::new(domain(1));
    let (identity, first) = identity
        .register_device_model_only(correlation(4), DeviceGenerationV1(1))
        .unwrap();
    let (identity, second) = identity
        .register_device_model_only(correlation(5), DeviceGenerationV1(1))
        .unwrap();
    let (identity, first_vm) = identity
        .register_vm_model_only(first, vm_observation(first, 10))
        .unwrap();
    let (_, second_vm) = identity
        .register_vm_model_only(second, vm_observation(second, 11))
        .unwrap();
    AdmissionFixture {
        first,
        second,
        first_vm,
        second_vm,
    }
}

fn mixed_generations_of_one_physical_device() -> (
    ModelVmAdmissionV1,
    ModelDeviceAdmissionV1,
    ModelDeviceAdmissionV1,
) {
    let identity = DeviceIdentityStateV1::new(domain(1));
    let correlated = correlation(4);
    let (identity, old_device) = identity
        .register_device_model_only(correlated, DeviceGenerationV1(1))
        .unwrap();
    let (identity, old_vm) = identity
        .register_vm_model_only(old_device, vm_observation(old_device, 90))
        .unwrap();
    let identity = identity.retire_vm_model_only(old_vm).unwrap();
    let identity = identity.retire_device_model_only(old_device).unwrap();
    let (_, new_device) = identity
        .register_device_model_only(correlated, DeviceGenerationV1(2))
        .unwrap();
    (old_vm, old_device, new_device)
}

fn advance(
    state: MemoryLifecycleStateV1,
    transition: MemoryTransitionV1,
) -> MemoryLifecycleStateV1 {
    let next = state.next(transition).unwrap();
    next.validate_global_invariants().unwrap();
    next
}

fn acquire(
    state: MemoryLifecycleStateV1,
    vm: ModelVmAdmissionV1,
    devices: Vec<ModelDeviceAdmissionV1>,
    handle: u64,
) -> MemoryLifecycleStateV1 {
    advance(
        state,
        MemoryTransitionV1::AcquireVm {
            admission: vm,
            mapping_devices: devices,
            handle: UntrustedVmHandleObservationV1(handle),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x10_0000,
            },
        },
    )
}

fn reservation(vm: VmKeyV1, id: u64) -> VaReservationKeyV1 {
    VaReservationKeyV1 {
        vm,
        id: VaReservationIdV1(id),
    }
}

fn allocation(vm: VmKeyV1, id: u64, generation: u64) -> MemoryAllocationKeyV1 {
    MemoryAllocationKeyV1 {
        vm,
        id: AllocationIdV1(id),
        generation: AllocationGenerationV1(generation),
    }
}

fn mapping(allocation: MemoryAllocationKeyV1, id: u64) -> MemoryMappingKeyV1 {
    MemoryMappingKeyV1 {
        allocation,
        id: MappingIdV1(id),
    }
}

fn spec() -> MemoryAllocationSpecV1 {
    MemoryAllocationSpecV1 {
        byte_len: MEMORY_PAGE_BYTES_V1,
        alignment: MEMORY_PAGE_BYTES_V1,
        kind: MemoryKindV1::HostVisibleCoherent,
        coherence: MemoryCoherenceV1::HostCoherent,
    }
}

fn live_allocation(
    devices: AdmissionFixture,
) -> (
    MemoryLifecycleStateV1,
    VaReservationKeyV1,
    MemoryAllocationKeyV1,
) {
    let vm = devices.first_vm.model_key();
    let reservation = reservation(vm, 20);
    let allocation = allocation(vm, 30, 1);
    let state = acquire(
        MemoryLifecycleStateV1::new(domain(1)),
        devices.first_vm,
        vec![devices.first, devices.second],
        100,
    );
    let state = advance(
        state,
        MemoryTransitionV1::ReserveVa {
            key: reservation,
            range: GpuVaRangeV1 {
                base: 0x2_0000,
                byte_len: MEMORY_PAGE_BYTES_V1,
            },
            alignment: MEMORY_PAGE_BYTES_V1,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::Allocate {
            key: allocation,
            reservation,
            handle: UntrustedAllocationHandleObservationV1(200),
            spec: spec(),
        },
    );
    (state, reservation, allocation)
}

#[test]
fn partial_map_and_unmap_retain_exact_device_progress_until_bottom_up_release() {
    let devices = admissions();
    let (state, reservation, allocation) = live_allocation(devices);
    let mapping = mapping(allocation, 40);
    let targets = vec![devices.first.model_key(), devices.second.model_key()];
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: mapping,
            target_devices: targets.clone(),
            access: MemoryAccessV1::ReadWrite,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(state.mappings()[0].state, MemoryMappingStateV1::MapFailed);
    assert_eq!(
        state.mappings()[0].retained_device_superset(),
        &targets[..1]
    );
    assert_eq!(
        state.next(MemoryTransitionV1::ReleaseAllocation { key: allocation }),
        Err(MemoryTransitionErrorV1::ResourceInUse(
            MemoryRecordRefV1::Allocation(allocation)
        ))
    );

    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 0,
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(
        state.mappings()[0].retained_device_superset(),
        &targets[..1]
    );
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    let state = advance(state, MemoryTransitionV1::ReleaseMapping { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ReleaseAllocation { key: allocation },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ReleaseVaReservation { key: reservation },
    );
    let state = advance(
        state,
        MemoryTransitionV1::RetireVm {
            key: devices.first_vm.model_key(),
        },
    );
    assert_eq!(state.vms()[0].state, MemoryVmStateV1::Retired);
    assert_eq!(state.reservations().len(), 1);
    assert_eq!(state.allocations().len(), 1);
    assert_eq!(state.mappings().len(), 1);
}

#[test]
fn publications_block_unmap_and_partial_unmap_retains_the_unreported_suffix() {
    let devices = admissions();
    let (state, _, allocation) = live_allocation(devices);
    let mapping = mapping(allocation, 40);
    let targets = vec![devices.first.model_key(), devices.second.model_key()];
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: mapping,
            target_devices: targets.clone(),
            access: MemoryAccessV1::Read,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 2,
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    let publication = MemoryPublicationKeyV1 {
        mapping,
        id: MemoryPublicationIdV1(50),
    };
    let state = advance(
        state,
        MemoryTransitionV1::PublishMapping { key: publication },
    );
    assert_eq!(
        state.publications()[0].owner,
        MemoryPublicationOwnerV1::Generic
    );
    assert_eq!(
        state.next(MemoryTransitionV1::BeginUnmap { key: mapping }),
        Err(MemoryTransitionErrorV1::ResourceInUse(
            MemoryRecordRefV1::Mapping(mapping)
        ))
    );
    let state = advance(
        state,
        MemoryTransitionV1::ReleasePublication { key: publication },
    );
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(
        state.mappings()[0].retained_device_superset(),
        &targets[1..]
    );
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseAllocation { key: allocation }),
        Err(MemoryTransitionErrorV1::ResourceInUse(_))
    ));
}

#[test]
fn unchanged_cumulative_unmap_retry_does_not_advance_twice() {
    let devices = admissions();
    let (state, _, allocation) = live_allocation(devices);
    let mapping = mapping(allocation, 42);
    let targets = vec![devices.first.model_key(), devices.second.model_key()];
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: mapping,
            target_devices: targets.clone(),
            access: MemoryAccessV1::ReadWrite,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: targets.len(),
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(state.mappings()[0].mapped_start, 1);
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(state.mappings()[0].mapped_start, 1);
    assert_eq!(
        state.mappings()[0].retained_device_superset(),
        &targets[1..]
    );
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseMapping { key: mapping }),
        Err(MemoryTransitionErrorV1::IllegalState(_))
    ));
}

#[test]
fn failed_full_cumulative_unmap_progress_remains_ambiguous_and_unreleasable() {
    let devices = admissions();
    let (state, _, allocation) = live_allocation(devices);
    let mapping = mapping(allocation, 43);
    let targets = vec![devices.first.model_key(), devices.second.model_key()];
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: mapping,
            target_devices: targets.clone(),
            access: MemoryAccessV1::ReadWrite,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: targets.len(),
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: targets.len(),
                status: PartialOperationStatusV1::Failed,
            },
        },
    );
    assert_eq!(state.mappings()[0].state, MemoryMappingStateV1::Ambiguous);
    assert_eq!(state.mappings()[0].mapped_start, 0);
    assert_eq!(state.mappings()[0].retained_device_superset(), targets);
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseMapping { key: mapping }),
        Err(MemoryTransitionErrorV1::IllegalState(_))
    ));
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseAllocation { key: allocation }),
        Err(MemoryTransitionErrorV1::ResourceInUse(_))
    ));
}

#[test]
fn malformed_or_indeterminate_progress_is_fail_closed_and_unreleasable() {
    let devices = admissions();
    let (state, _, allocation) = live_allocation(devices);
    let ambiguous_map = mapping(allocation, 40);
    let targets = vec![devices.first.model_key(), devices.second.model_key()];
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: ambiguous_map,
            target_devices: targets.clone(),
            access: MemoryAccessV1::ReadWrite,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: ambiguous_map,
            progress: PartialProgressObservationV1 {
                n_success: 1,
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    assert_eq!(state.mappings()[0].state, MemoryMappingStateV1::Ambiguous);
    assert_eq!(state.mappings()[0].retained_device_superset(), targets);
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseMapping { key: ambiguous_map }),
        Err(MemoryTransitionErrorV1::IllegalState(_))
    ));
    assert!(matches!(
        state.next(MemoryTransitionV1::ReleaseAllocation { key: allocation }),
        Err(MemoryTransitionErrorV1::ResourceInUse(_))
    ));

    let (state, _, allocation) = live_allocation(devices);
    let mapping = mapping(allocation, 41);
    let state = advance(
        state,
        MemoryTransitionV1::BeginMap {
            key: mapping,
            target_devices: targets.clone(),
            access: MemoryAccessV1::ReadWrite,
        },
    );
    let state = advance(
        state,
        MemoryTransitionV1::ObserveMap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: targets.len(),
                status: PartialOperationStatusV1::Succeeded,
            },
        },
    );
    let state = advance(state, MemoryTransitionV1::BeginUnmap { key: mapping });
    let state = advance(
        state,
        MemoryTransitionV1::ObserveUnmap {
            key: mapping,
            progress: PartialProgressObservationV1 {
                n_success: targets.len() + 1,
                status: PartialOperationStatusV1::Indeterminate,
            },
        },
    );
    assert_eq!(state.mappings()[0].state, MemoryMappingStateV1::Ambiguous);
    assert_eq!(state.mappings()[0].retained_device_superset(), targets);
}

#[test]
fn ranges_device_sets_and_cross_vm_substitutions_reject_failure_atomically() {
    let devices = admissions();
    let (state, _, live_allocation_key) = live_allocation(devices);
    let before = state.clone();
    let wrong_mapping = mapping(live_allocation_key, 41);
    assert_eq!(
        state.next(MemoryTransitionV1::BeginMap {
            key: wrong_mapping,
            target_devices: vec![devices.second.model_key()],
            access: MemoryAccessV1::Read,
        }),
        Err(MemoryTransitionErrorV1::DeviceSetMismatch(
            MemoryRecordRefV1::Mapping(wrong_mapping)
        ))
    );
    assert_eq!(state, before);

    let (stale_vm, old_device, new_device) = mixed_generations_of_one_physical_device();
    assert_eq!(
        MemoryLifecycleStateV1::new(domain(1)).next(MemoryTransitionV1::AcquireVm {
            admission: stale_vm,
            mapping_devices: vec![old_device, new_device],
            handle: UntrustedVmHandleObservationV1(102),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x10_0000,
            },
        }),
        Err(MemoryTransitionErrorV1::DeviceSetMismatch(
            MemoryRecordRefV1::Vm(stale_vm.model_key())
        ))
    );

    let state = acquire(
        state,
        devices.second_vm,
        vec![devices.first, devices.second],
        101,
    );
    let second_reservation = reservation(devices.second_vm.model_key(), 21);
    let state = advance(
        state,
        MemoryTransitionV1::ReserveVa {
            key: second_reservation,
            range: GpuVaRangeV1 {
                base: 0x2_0000,
                byte_len: MEMORY_PAGE_BYTES_V1,
            },
            alignment: MEMORY_PAGE_BYTES_V1,
        },
    );
    let cross_vm = allocation(devices.first_vm.model_key(), 31, 1);
    assert_eq!(
        state.next(MemoryTransitionV1::Allocate {
            key: cross_vm,
            reservation: second_reservation,
            handle: UntrustedAllocationHandleObservationV1(201),
            spec: spec(),
        }),
        Err(MemoryTransitionErrorV1::BindingMismatch(
            MemoryRecordRefV1::Allocation(cross_vm)
        ))
    );

    let overlap = reservation(devices.first_vm.model_key(), 22);
    assert_eq!(
        state.next(MemoryTransitionV1::ReserveVa {
            key: overlap,
            range: GpuVaRangeV1 {
                base: 0x2_0000,
                byte_len: MEMORY_PAGE_BYTES_V1,
            },
            alignment: MEMORY_PAGE_BYTES_V1,
        }),
        Err(MemoryTransitionErrorV1::AddressConflict(overlap))
    );
    let overflow = reservation(devices.first_vm.model_key(), 23);
    assert!(matches!(
        state.next(MemoryTransitionV1::ReserveVa {
            key: overflow,
            range: GpuVaRangeV1 {
                base: u64::MAX - (MEMORY_PAGE_BYTES_V1 - 1),
                byte_len: MEMORY_PAGE_BYTES_V1,
            },
            alignment: MEMORY_PAGE_BYTES_V1,
        }),
        Err(MemoryTransitionErrorV1::InvalidRange(_))
    ));
    let misaligned = reservation(devices.first_vm.model_key(), 24);
    assert!(matches!(
        state.next(MemoryTransitionV1::ReserveVa {
            key: misaligned,
            range: GpuVaRangeV1 {
                base: 0x4_0001,
                byte_len: MEMORY_PAGE_BYTES_V1,
            },
            alignment: MEMORY_PAGE_BYTES_V1,
        }),
        Err(MemoryTransitionErrorV1::InvalidAlignment(_))
    ));
}

#[test]
fn allocation_generations_and_opaque_handles_cannot_be_substituted() {
    let devices = admissions();
    let vm = devices.first_vm.model_key();
    let first_reservation = reservation(vm, 20);
    let second_reservation = reservation(vm, 21);
    let mut state = acquire(
        MemoryLifecycleStateV1::new(domain(1)),
        devices.first_vm,
        vec![devices.first],
        100,
    );
    for (key, base) in [
        (first_reservation, 0x2_0000),
        (second_reservation, 0x3_0000),
    ] {
        state = advance(
            state,
            MemoryTransitionV1::ReserveVa {
                key,
                range: GpuVaRangeV1 {
                    base,
                    byte_len: MEMORY_PAGE_BYTES_V1,
                },
                alignment: MEMORY_PAGE_BYTES_V1,
            },
        );
    }
    let generation_two = allocation(vm, 30, 2);
    state = advance(
        state,
        MemoryTransitionV1::Allocate {
            key: generation_two,
            reservation: first_reservation,
            handle: UntrustedAllocationHandleObservationV1(200),
            spec: spec(),
        },
    );
    let colliding = allocation(vm, 31, 1);
    assert_eq!(
        state.next(MemoryTransitionV1::Allocate {
            key: colliding,
            reservation: second_reservation,
            handle: UntrustedAllocationHandleObservationV1(200),
            spec: spec(),
        }),
        Err(MemoryTransitionErrorV1::HandleCollision(
            MemoryRecordRefV1::Allocation(colliding)
        ))
    );
    state = advance(
        state,
        MemoryTransitionV1::ReleaseAllocation {
            key: generation_two,
        },
    );
    let stale = allocation(vm, 30, 1);
    assert_eq!(
        state.next(MemoryTransitionV1::Allocate {
            key: stale,
            reservation: first_reservation,
            handle: UntrustedAllocationHandleObservationV1(200),
            spec: spec(),
        }),
        Err(MemoryTransitionErrorV1::StaleGeneration(stale))
    );
    let generation_three = allocation(vm, 30, 3);
    state = advance(
        state,
        MemoryTransitionV1::Allocate {
            key: generation_three,
            reservation: first_reservation,
            handle: UntrustedAllocationHandleObservationV1(200),
            spec: spec(),
        },
    );
    assert_eq!(state.allocations().len(), 2);
    assert_eq!(state.allocations()[1].key, generation_three);
}

#[test]
fn vm_history_has_a_process_lifetime_bound_and_rejects_domain_substitution() {
    let domain_id = domain(1);
    let identity = DeviceIdentityStateV1::new(domain_id);
    let (mut identity, device) = identity
        .register_device_model_only(correlation(4), DeviceGenerationV1(1))
        .unwrap();
    let mut memory = MemoryLifecycleStateV1::new(domain_id);
    let mut first_vm = None;
    for id in 1..=MAX_MEMORY_VMS_V1 as u64 {
        let (next_identity, vm) = identity
            .register_vm_model_only(device, vm_observation(device, id))
            .unwrap();
        identity = next_identity;
        first_vm.get_or_insert(vm);
        memory = acquire(memory, vm, vec![device], 1_000 + id);
    }
    let overflow = first_vm.unwrap();
    assert_eq!(
        memory.next(MemoryTransitionV1::AcquireVm {
            admission: overflow,
            mapping_devices: vec![device],
            handle: UntrustedVmHandleObservationV1(9_999),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x10_0000,
            },
        }),
        Err(MemoryTransitionErrorV1::CapacityExceeded {
            kind: MemoryRecordKindV1::Vm,
            maximum: MAX_MEMORY_VMS_V1,
        })
    );
    assert_eq!(memory.vms().len(), MAX_MEMORY_VMS_V1);

    let foreign_memory = MemoryLifecycleStateV1::new(domain(9));
    assert_eq!(
        foreign_memory.next(MemoryTransitionV1::AcquireVm {
            admission: overflow,
            mapping_devices: vec![device],
            handle: UntrustedVmHandleObservationV1(9_999),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x10_0000,
            },
        }),
        Err(MemoryTransitionErrorV1::ObservationDomainMismatch)
    );
}
