use super::*;
use fe2o3_runtime_model as model;

fn digest(seed: u8) -> model::IdentityDigestV1 {
    model::IdentityDigestV1::from_untrusted_bytes([seed; model::IDENTITY_DIGEST_BYTES_V1])
}

fn domain() -> model::DeviceObservationDomainIdV1 {
    model::DeviceObservationDomainIdV1::from_untrusted_digest(digest(1))
}

fn profile() -> model::DeviceAdmissionProfileV1 {
    model::DeviceAdmissionProfileV1::gfx942_xnack_minus_spx_nps1_kfd_1_18_drm_3_64_0(
        model::DeviceAdmissionProfileIdV1::from_untrusted_digest(digest(2)),
        digest(3),
        digest(4),
    )
}

fn correlation() -> model::ModelCorrelatedDeviceV1 {
    let domain_id = domain();
    let epoch = model::ObservationEpochV1(9);
    let pci = model::PciAddressV1 {
        domain: 0,
        bus: 1,
        device: 1,
        function: 0,
    };
    model::UntrustedDeviceInventoryV1::from_untrusted_observations(
        model::UntrustedKfdObservationV1 {
            domain_id,
            epoch,
            node: model::DeviceNodeV1 {
                major: 511,
                minor: model::KFD_DEVICE_MINOR_V1,
            },
            uapi_major: model::KFD_UAPI_MAJOR_V1,
            uapi_minor: model::KFD_UAPI_MINOR_V1,
            schema_identity: digest(3),
            xnack: model::XnackObservationV1::Disabled,
        },
        vec![model::UntrustedTopologyObservationV1 {
            domain_id,
            epoch,
            topology_node_id: 1,
            kfd_gpu_id: 7,
            gpu_unique_id: 101,
            drm_render_minor: model::DRM_RENDER_MIN_MINOR_V1 + 1,
            pci,
            vendor_id: model::AMD_PCI_VENDOR_ID_V1,
            device_id: model::MI300X_PCI_DEVICE_ID_V1,
            target: model::GpuTargetObservationV1::Gfx942,
            compute_partition: model::ComputePartitionObservationV1::Spx,
            memory_partition: model::MemoryPartitionObservationV1::Nps1,
        }],
        vec![model::UntrustedRenderObservationV1 {
            domain_id,
            epoch,
            node: model::DeviceNodeV1 {
                major: model::DRM_DEVICE_MAJOR_V1,
                minor: model::DRM_RENDER_MIN_MINOR_V1 + 1,
            },
            gpu_unique_id: 101,
            pci,
            vendor_id: model::AMD_PCI_VENDOR_ID_V1,
            device_id: model::MI300X_PCI_DEVICE_ID_V1,
            pci_revision_id: 0,
            drm_schema_identity: digest(4),
            driver_name: model::DrmDriverNameObservationV1::Amdgpu,
            drm_major: model::DRM_DRIVER_MAJOR_V1,
            drm_minor: model::DRM_DRIVER_MINOR_V1,
            drm_patch: model::DRM_DRIVER_PATCH_V1,
            acceleration_working: true,
            family: model::DrmFamilyObservationV1::AmdgpuFamilyAi,
        }],
    )
    .unwrap()
    .correlate_model_only(&profile())
    .unwrap()
}

struct JournalFixture {
    state: MemoryLifecycleStateV1,
    device: ModelDeviceAdmissionV1,
    reservation: VaReservationKeyV1,
    allocation: MemoryAllocationKeyV1,
    mapping: MemoryMappingKeyV1,
}

fn journal_fixture() -> JournalFixture {
    let domain_id = domain();
    let (identities, device) = model::DeviceIdentityStateV1::new(domain_id)
        .register_device_model_only(correlation(), model::DeviceGenerationV1(1))
        .unwrap();
    let correlated = device.correlation();
    let (_, vm) = identities
        .register_vm_model_only(
            device,
            model::UntrustedVmObservationV1 {
                domain_id,
                device: device.model_key(),
                vm_id: VmIdV1(1),
                kfd_gpu_id: correlated.kfd_gpu_id(),
                render_node: correlated.render_node(),
                pci: correlated.identity().pci,
            },
        )
        .unwrap();
    let state = MemoryLifecycleStateV1::new(domain_id)
        .next(MemoryTransitionV1::AcquireVm {
            admission: vm,
            mapping_devices: vec![device],
            handle: UntrustedVmHandleObservationV1(1),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x20_0000,
            },
        })
        .unwrap();
    let reservation = VaReservationKeyV1 {
        vm: vm.model_key(),
        id: VaReservationIdV1(1),
    };
    let allocation = MemoryAllocationKeyV1 {
        vm: vm.model_key(),
        id: AllocationIdV1(1),
        generation: AllocationGenerationV1(1),
    };
    JournalFixture {
        state,
        device,
        reservation,
        allocation,
        mapping: MemoryMappingKeyV1 {
            allocation,
            id: MappingIdV1(1),
        },
    }
}

struct JournalBackend;

impl MemoryBackend for JournalBackend {
    type Reservation = ();
    type Mapping = ();

    fn opener_pid(&self) -> u32 {
        std::process::id()
    }

    fn gpu_id(&self) -> u32 {
        7
    }

    fn gpuvm_aperture(&self) -> InclusiveAperture {
        InclusiveAperture::from_checked_parts_for_memory_tests(0x1_0000, 0x20_ffff)
    }

    fn page_size(&self) -> usize {
        4096
    }

    fn check_currentness(&mut self) -> Result<(), MemorySessionError> {
        Ok(())
    }

    fn acquire_vm(&mut self) -> Result<(), MemorySessionError> {
        Ok(())
    }

    fn reserve_va(&mut self, _bytes: usize) -> Result<(), MemorySessionError> {
        Err(MemorySessionError::Injected("unused reserve"))
    }

    fn reservation_address(_reservation: &()) -> u64 {
        0
    }

    fn alloc(&mut self, _va: u64, _bytes: u64) -> KernelOutcome<KfdIoctlAllocMemoryOfGpuArgs> {
        panic!("unused alloc")
    }

    fn map_cpu(
        &mut self,
        _reservation: &mut (),
        _mmap_offset: u64,
        _bytes: usize,
    ) -> Result<(), MemorySessionError> {
        Err(MemorySessionError::Injected("unused mmap"))
    }

    fn prepare_cpu_mapping(&mut self, _mapping: &mut ()) -> Result<(), MemorySessionError> {
        Err(MemorySessionError::Injected("unused setup"))
    }

    fn map_gpu(&mut self, _handle: u64, _old_success: u32) -> KernelOutcome<u32> {
        panic!("unused map")
    }

    fn unmap_gpu(&mut self, _handle: u64, _old_success: u32) -> KernelOutcome<u32> {
        panic!("unused unmap")
    }

    fn with_bytes<R>(_mapping: &(), _requested_bytes: usize, _f: impl FnOnce(&[u8]) -> R) -> R {
        panic!("unused borrow")
    }

    fn with_bytes_mut<R>(
        _mapping: &mut (),
        _requested_bytes: usize,
        _f: impl FnOnce(&mut [u8]) -> R,
    ) -> R {
        panic!("unused mutable borrow")
    }

    fn unmap_cpu(&mut self, _mapping: &mut ()) -> Result<(), MemorySessionError> {
        Err(MemorySessionError::Injected("unused munmap"))
    }

    fn free(&mut self, _handle: u64) -> Result<(), MemorySessionError> {
        Err(MemorySessionError::Injected("unused free"))
    }
}

#[test]
fn private_session_completion_journal_has_exact_order_counts_and_states() {
    let fixture = journal_fixture();
    let mut state = fixture.state;
    assert_eq!(state.vms().len(), 1);
    assert!(state.reservations().is_empty());
    assert!(state.allocations().is_empty());
    assert!(state.mappings().is_empty());

    state = project_allocation_completion(
        &state,
        fixture.reservation,
        fixture.allocation,
        0x2_0000,
        HostVisibleAllocationLayout {
            requested_bytes: 4096,
            backing_bytes: 4096,
        },
        0x55,
    )
    .unwrap();
    assert_eq!(state.reservations().len(), 1);
    assert_eq!(state.allocations().len(), 1);
    assert_eq!(
        state.reservations()[0].state,
        model::VaReservationStateV1::Reserved
    );
    assert_eq!(
        state.allocations()[0].state,
        model::MemoryAllocationStateV1::Live
    );
    assert!(state.mappings().is_empty());

    state = project_map_completion(&state, fixture.mapping, fixture.device).unwrap();
    assert_eq!(state.mappings().len(), 1);
    assert_eq!(
        state.mappings()[0].state,
        model::MemoryMappingStateV1::Mapped
    );
    assert_eq!(state.mappings()[0].retained_device_superset().len(), 1);

    state = project_unmap_completion(&state, fixture.mapping).unwrap();
    assert_eq!(state.mappings().len(), 1);
    assert_eq!(
        state.mappings()[0].state,
        model::MemoryMappingStateV1::Unmapped
    );
    assert!(state.mappings()[0].retained_device_superset().is_empty());

    state = project_release_completion(
        &state,
        fixture.mapping,
        fixture.allocation,
        fixture.reservation,
    )
    .unwrap();
    assert_eq!(state.reservations().len(), 1);
    assert_eq!(state.allocations().len(), 1);
    assert_eq!(state.mappings().len(), 1);
    assert_eq!(
        state.mappings()[0].state,
        model::MemoryMappingStateV1::Released
    );
    assert_eq!(
        state.allocations()[0].state,
        model::MemoryAllocationStateV1::Released
    );
    assert_eq!(
        state.reservations()[0].state,
        model::VaReservationStateV1::Released
    );
}

#[test]
fn projection_failure_quarantines_the_adapter_engine() {
    let fixture = journal_fixture();
    let mut invalid = MemoryLifecycleStateV1::new(fixture.state.domain_id());
    let projected = project_allocation_completion(
        &invalid,
        fixture.reservation,
        fixture.allocation,
        0x2_0000,
        HostVisibleAllocationLayout {
            requested_bytes: 4096,
            backing_bytes: 4096,
        },
        0x55,
    );
    assert!(projected.is_err());
    let mut engine = MemoryEngine::acquire(JournalBackend).unwrap();
    assert!(
        commit_model_projection(
            &mut engine,
            &mut invalid,
            projected,
            "injected projection failure",
        )
        .is_err()
    );
    assert_eq!(engine.phase(), HostVisibleMemoryPhase::Quarantined);
    assert!(invalid.vms().is_empty());
}
