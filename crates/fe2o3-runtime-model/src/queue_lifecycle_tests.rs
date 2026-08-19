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

fn correlation() -> ModelCorrelatedDeviceV1 {
    let domain_id = domain(1);
    let epoch = ObservationEpochV1(9);
    let pci = PciAddressV1 {
        domain: 0,
        bus: 5,
        device: 0,
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
            topology_node_id: 2,
            kfd_gpu_id: 28_851,
            gpu_unique_id: 0x6ced_1647_a296_545c,
            drm_render_minor: DRM_RENDER_MIN_MINOR_V1,
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
                minor: DRM_RENDER_MIN_MINOR_V1,
            },
            gpu_unique_id: 0x6ced_1647_a296_545c,
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

fn vm_observation(device: ModelDeviceAdmissionV1) -> UntrustedVmObservationV1 {
    let correlated = device.correlation();
    UntrustedVmObservationV1 {
        domain_id: correlated.domain_id(),
        device: device.model_key(),
        vm_id: VmIdV1(10),
        kfd_gpu_id: correlated.kfd_gpu_id(),
        render_node: correlated.render_node(),
        pci: correlated.identity().pci,
    }
}

fn memory_advance(
    state: MemoryLifecycleStateV1,
    transition: MemoryTransitionV1,
) -> MemoryLifecycleStateV1 {
    state.next(transition).unwrap()
}

struct QueueFixture {
    identity: DeviceIdentityStateV1,
    device: ModelDeviceAdmissionV1,
    vm: ModelVmAdmissionV1,
    memory: MemoryLifecycleStateV1,
    plan: ComputeAqlQueuePlanV1,
}

fn fixture() -> QueueFixture {
    let identity = DeviceIdentityStateV1::new(domain(1));
    let (identity, device) = identity
        .register_device_model_only(correlation(), DeviceGenerationV1(1))
        .unwrap();
    let (identity, vm) = identity
        .register_vm_model_only(device, vm_observation(device))
        .unwrap();
    let vm_key = vm.model_key();
    let mut memory = memory_advance(
        MemoryLifecycleStateV1::new(domain(1)),
        MemoryTransitionV1::AcquireVm {
            admission: vm,
            mapping_devices: vec![device],
            handle: UntrustedVmHandleObservationV1(100),
            aperture: GpuVaRangeV1 {
                base: 0x1_0000,
                byte_len: 0x20_0000,
            },
        },
    );

    let mut bindings = Vec::new();
    for index in 0_u64..COMPUTE_AQL_RESOURCE_COUNT_V1 as u64 {
        let reservation = VaReservationKeyV1 {
            vm: vm_key,
            id: VaReservationIdV1(200 + index),
        };
        let allocation = MemoryAllocationKeyV1 {
            vm: vm_key,
            id: AllocationIdV1(300 + index),
            generation: AllocationGenerationV1(1),
        };
        let mapping = MemoryMappingKeyV1 {
            allocation,
            id: MappingIdV1(400 + index),
        };
        memory = memory_advance(
            memory,
            MemoryTransitionV1::ReserveVa {
                key: reservation,
                range: GpuVaRangeV1 {
                    base: 0x2_0000 + index * MEMORY_PAGE_BYTES_V1,
                    byte_len: MEMORY_PAGE_BYTES_V1,
                },
                alignment: MEMORY_PAGE_BYTES_V1,
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::Allocate {
                key: allocation,
                reservation,
                handle: UntrustedAllocationHandleObservationV1(500 + index),
                spec: MemoryAllocationSpecV1 {
                    byte_len: MEMORY_PAGE_BYTES_V1,
                    alignment: MEMORY_PAGE_BYTES_V1,
                    kind: MemoryKindV1::QueueStorage,
                    coherence: MemoryCoherenceV1::HostCoherent,
                },
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::BeginMap {
                key: mapping,
                target_devices: vec![device.model_key()],
                access: MemoryAccessV1::ReadWrite,
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::ObserveMap {
                key: mapping,
                progress: PartialProgressObservationV1 {
                    n_success: 1,
                    status: PartialOperationStatusV1::Succeeded,
                },
            },
        );
        bindings.push(ComputeAqlResourceBindingV1 {
            mapping,
            publication: MemoryPublicationKeyV1 {
                mapping,
                id: MemoryPublicationIdV1(600 + index),
            },
            expected_kind: MemoryKindV1::QueueStorage,
            expected_coherence: MemoryCoherenceV1::HostCoherent,
            expected_access: MemoryAccessV1::ReadWrite,
        });
    }
    let resources = ComputeAqlQueueResourcesV1 {
        ring: bindings[0],
        control: bindings[1],
        eop: bindings[2],
        context_save: bindings[3],
    };
    let plan = ComputeAqlQueuePlanV1 {
        schema_version: QUEUE_LIFECYCLE_SCHEMA_VERSION_V1,
        target: ComputeAqlTargetProfileV1::Gfx942XnackMinusSpxNps1Kfd1_18,
        domain_id: domain(1),
        plan_id: QueuePlanIdV1::from_untrusted_digest(digest(5)),
        current_device: device,
        queue: QueueKeyV1 {
            vm: vm_key,
            id: QueueInstanceIdV1(700),
            generation: QueueGenerationV1(1),
        },
        initial_configuration: QueueConfigurationIdV1::from_untrusted_digest(digest(6)),
        resources,
    };
    QueueFixture {
        identity,
        device,
        vm,
        memory,
        plan,
    }
}

fn append_resource_set(
    mut memory: MemoryLifecycleStateV1,
    fixture: &QueueFixture,
    identity_base: u64,
    va_base: u64,
) -> (MemoryLifecycleStateV1, ComputeAqlQueueResourcesV1) {
    let mut bindings = Vec::new();
    for index in 0_u64..COMPUTE_AQL_RESOURCE_COUNT_V1 as u64 {
        let reservation = VaReservationKeyV1 {
            vm: fixture.vm.model_key(),
            id: VaReservationIdV1(identity_base + index),
        };
        let allocation = MemoryAllocationKeyV1 {
            vm: fixture.vm.model_key(),
            id: AllocationIdV1(identity_base + 100 + index),
            generation: AllocationGenerationV1(1),
        };
        let mapping = MemoryMappingKeyV1 {
            allocation,
            id: MappingIdV1(identity_base + 200 + index),
        };
        memory = memory_advance(
            memory,
            MemoryTransitionV1::ReserveVa {
                key: reservation,
                range: GpuVaRangeV1 {
                    base: va_base + index * MEMORY_PAGE_BYTES_V1,
                    byte_len: MEMORY_PAGE_BYTES_V1,
                },
                alignment: MEMORY_PAGE_BYTES_V1,
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::Allocate {
                key: allocation,
                reservation,
                handle: UntrustedAllocationHandleObservationV1(identity_base + 300 + index),
                spec: MemoryAllocationSpecV1 {
                    byte_len: MEMORY_PAGE_BYTES_V1,
                    alignment: MEMORY_PAGE_BYTES_V1,
                    kind: MemoryKindV1::QueueStorage,
                    coherence: MemoryCoherenceV1::HostCoherent,
                },
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::BeginMap {
                key: mapping,
                target_devices: vec![fixture.device.model_key()],
                access: MemoryAccessV1::ReadWrite,
            },
        );
        memory = memory_advance(
            memory,
            MemoryTransitionV1::ObserveMap {
                key: mapping,
                progress: PartialProgressObservationV1 {
                    n_success: 1,
                    status: PartialOperationStatusV1::Succeeded,
                },
            },
        );
        bindings.push(ComputeAqlResourceBindingV1 {
            mapping,
            publication: MemoryPublicationKeyV1 {
                mapping,
                id: MemoryPublicationIdV1(identity_base + 400 + index),
            },
            expected_kind: MemoryKindV1::QueueStorage,
            expected_coherence: MemoryCoherenceV1::HostCoherent,
            expected_access: MemoryAccessV1::ReadWrite,
        });
    }
    (
        memory,
        ComputeAqlQueueResourcesV1 {
            ring: bindings[0],
            control: bindings[1],
            eop: bindings[2],
            context_save: bindings[3],
        },
    )
}

fn distinct_plan(
    fixture: &QueueFixture,
    resources: ComputeAqlQueueResourcesV1,
    instance: u64,
    generation: u64,
    seed: u8,
) -> ComputeAqlQueuePlanV1 {
    ComputeAqlQueuePlanV1 {
        schema_version: QUEUE_LIFECYCLE_SCHEMA_VERSION_V1,
        target: ComputeAqlTargetProfileV1::Gfx942XnackMinusSpxNps1Kfd1_18,
        domain_id: domain(1),
        plan_id: QueuePlanIdV1::from_untrusted_digest(digest(seed)),
        current_device: fixture.device,
        queue: QueueKeyV1 {
            vm: fixture.vm.model_key(),
            id: QueueInstanceIdV1(instance),
            generation: QueueGenerationV1(generation),
        },
        initial_configuration: QueueConfigurationIdV1::from_untrusted_digest(digest(seed + 1)),
        resources,
    }
}

fn admit(fixture: &QueueFixture) -> QueuePlanAdmissionV1 {
    QueueLifecycleStateV1::new(domain(1))
        .admit_compute_aql_plan(&fixture.identity, &fixture.memory, fixture.plan)
        .unwrap()
}

fn advance(
    queue: QueueLifecycleStateV1,
    fixture: &QueueFixture,
    memory: &MemoryLifecycleStateV1,
    transition: QueueTransitionV1,
) -> QueueLifecycleStateV1 {
    queue.next(&fixture.identity, memory, transition).unwrap()
}

fn create_active(
    queue: QueueLifecycleStateV1,
    fixture: &QueueFixture,
    memory: &MemoryLifecycleStateV1,
) -> QueueLifecycleStateV1 {
    let key = fixture.plan.queue;
    let queue = advance(
        queue,
        fixture,
        memory,
        QueueTransitionV1::BeginCreate { queue: key },
    );
    advance(
        queue,
        fixture,
        memory,
        QueueTransitionV1::ObserveCreate {
            queue: key,
            observation: QueueCreateObservationV1 {
                status: QueueSyscallStatusV1::Succeeded,
                queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                    UntrustedQueueIdObservationV1(23),
                ),
            },
        },
    )
}

#[test]
fn exact_plan_publishes_and_retains_all_four_mapped_resources() {
    let fixture = fixture();
    let admission = admit(&fixture);
    assert_eq!(admission.authority_domain(), AuthorityDomainV1::ModelOnly);
    assert_eq!(admission.memory_state().publications().len(), 4);
    assert!(
        admission
            .memory_state()
            .publications()
            .iter()
            .all(|record| record.state == MemoryPublicationStateV1::Live)
    );
    for (_, resource) in fixture.plan.resources.ordered() {
        assert!(
            !admission
                .queue_state()
                .can_release_mapping(resource.mapping)
        );
        assert_eq!(
            admission
                .memory_state()
                .next(MemoryTransitionV1::BeginUnmap {
                    key: resource.mapping,
                }),
            Err(MemoryTransitionErrorV1::ResourceInUse(
                MemoryRecordRefV1::Mapping(resource.mapping)
            ))
        );
    }
}

#[test]
fn successful_lifecycle_retains_queue_identity_and_append_only_history() {
    let fixture = fixture();
    let (mut queue, memory) = admit(&fixture).into_states();
    let key = fixture.plan.queue;
    let mut old_history = queue.history().to_vec();
    queue = create_active(queue, &fixture, &memory);
    assert_eq!(
        &queue.history()[..old_history.len()],
        old_history.as_slice()
    );
    old_history = queue.history().to_vec();

    let next_configuration = QueueConfigurationIdV1::from_untrusted_digest(digest(7));
    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginUpdate {
            queue: key,
            configuration: next_configuration,
        },
    );
    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveUpdate {
            queue: key,
            status: QueueSyscallStatusV1::Succeeded,
        },
    );
    assert_eq!(
        &queue.history()[..old_history.len()],
        old_history.as_slice()
    );
    assert_eq!(queue.queues()[0].configuration, next_configuration);

    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginDisable { queue: key },
    );
    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveDisable {
            queue: key,
            status: QueueSyscallStatusV1::Succeeded,
        },
    );
    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginDestroy { queue: key },
    );
    queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveDestroy {
            queue: key,
            status: QueueSyscallStatusV1::Succeeded,
        },
    );
    let record = queue.queues()[0];
    assert_eq!(record.phase, ComputeAqlQueuePhaseV1::Destroyed);
    assert_eq!(record.queue_id, Some(UntrustedQueueIdObservationV1(23)));
    assert_eq!(record.plan.queue.generation, QueueGenerationV1(1));
    for (_, resource) in fixture.plan.resources.ordered() {
        assert!(queue.can_release_mapping(resource.mapping));
    }
    let memory = queue.release_resource_publications(&memory, key).unwrap();
    assert!(
        memory
            .publications()
            .iter()
            .all(|record| record.state == MemoryPublicationStateV1::Released)
    );
}

#[test]
fn malformed_or_indeterminate_create_is_terminal_and_unreleasable() {
    for observation in [
        QueueCreateObservationV1 {
            status: QueueSyscallStatusV1::Succeeded,
            queue_id_field: CreateQueueIdFieldObservationV1::SentinelUnchanged,
        },
        QueueCreateObservationV1 {
            status: QueueSyscallStatusV1::Succeeded,
            queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                UntrustedQueueIdObservationV1(CREATE_QUEUE_ID_SENTINEL_V1),
            ),
        },
        QueueCreateObservationV1 {
            status: QueueSyscallStatusV1::Indeterminate,
            queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                UntrustedQueueIdObservationV1(91),
            ),
        },
        QueueCreateObservationV1 {
            status: QueueSyscallStatusV1::FailedNoEffect,
            queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                UntrustedQueueIdObservationV1(92),
            ),
        },
    ] {
        let fixture = fixture();
        let (queue, memory) = admit(&fixture).into_states();
        let key = fixture.plan.queue;
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::BeginCreate { queue: key },
        );
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::ObserveCreate {
                queue: key,
                observation,
            },
        );
        assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Ambiguous);
        assert!(queue.release_resource_publications(&memory, key).is_err());
        assert!(
            fixture
                .plan
                .resources
                .ordered()
                .iter()
                .all(|(_, resource)| !queue.can_release_mapping(resource.mapping))
        );
    }
}

#[test]
fn returned_queue_id_zero_is_valid_and_unchanged_sentinel_is_not_an_id() {
    let zero_id_fixture = fixture();
    let (queue, memory) = admit(&zero_id_fixture).into_states();
    let key = zero_id_fixture.plan.queue;
    let queue = advance(
        queue,
        &zero_id_fixture,
        &memory,
        QueueTransitionV1::BeginCreate { queue: key },
    );
    let queue = advance(
        queue,
        &zero_id_fixture,
        &memory,
        QueueTransitionV1::ObserveCreate {
            queue: key,
            observation: QueueCreateObservationV1 {
                status: QueueSyscallStatusV1::Succeeded,
                queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                    UntrustedQueueIdObservationV1(0),
                ),
            },
        },
    );
    assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Active);
    assert_eq!(
        queue.queues()[0].queue_id,
        Some(UntrustedQueueIdObservationV1(0))
    );

    let second_fixture = fixture();
    let (queue, memory) = admit(&second_fixture).into_states();
    let queue = advance(
        queue,
        &second_fixture,
        &memory,
        QueueTransitionV1::BeginCreate { queue: key },
    );
    let queue = advance(
        queue,
        &second_fixture,
        &memory,
        QueueTransitionV1::ObserveCreate {
            queue: key,
            observation: QueueCreateObservationV1 {
                status: QueueSyscallStatusV1::FailedNoEffect,
                queue_id_field: CreateQueueIdFieldObservationV1::SentinelUnchanged,
            },
        },
    );
    assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Planned);
    assert_eq!(queue.queues()[0].queue_id, None);
}

#[test]
fn indeterminate_update_disable_and_destroy_retain_every_resource() {
    for stage in 0..3 {
        let fixture = fixture();
        let (queue, memory) = admit(&fixture).into_states();
        let key = fixture.plan.queue;
        let mut queue = create_active(queue, &fixture, &memory);
        queue = match stage {
            0 => {
                let queue = advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::BeginUpdate {
                        queue: key,
                        configuration: QueueConfigurationIdV1::from_untrusted_digest(digest(7)),
                    },
                );
                advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::ObserveUpdate {
                        queue: key,
                        status: QueueSyscallStatusV1::Indeterminate,
                    },
                )
            }
            1 => {
                let queue = advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::BeginDisable { queue: key },
                );
                advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::ObserveDisable {
                        queue: key,
                        status: QueueSyscallStatusV1::Indeterminate,
                    },
                )
            }
            _ => {
                let queue = advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::BeginDisable { queue: key },
                );
                let queue = advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::ObserveDisable {
                        queue: key,
                        status: QueueSyscallStatusV1::Succeeded,
                    },
                );
                let queue = advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::BeginDestroy { queue: key },
                );
                advance(
                    queue,
                    &fixture,
                    &memory,
                    QueueTransitionV1::ObserveDestroy {
                        queue: key,
                        status: QueueSyscallStatusV1::Indeterminate,
                    },
                )
            }
        };
        assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Ambiguous);
        assert!(queue.release_resource_publications(&memory, key).is_err());
    }
}

#[test]
fn failed_no_effect_observations_restore_the_exact_prior_phase() {
    let fixture = fixture();
    let (queue, memory) = admit(&fixture).into_states();
    let key = fixture.plan.queue;
    let queue = create_active(queue, &fixture, &memory);
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginDisable { queue: key },
    );
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveDisable {
            queue: key,
            status: QueueSyscallStatusV1::FailedNoEffect,
        },
    );
    assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Active);

    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginDisable { queue: key },
    );
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveDisable {
            queue: key,
            status: QueueSyscallStatusV1::Succeeded,
        },
    );
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginUpdate {
            queue: key,
            configuration: QueueConfigurationIdV1::from_untrusted_digest(digest(7)),
        },
    );
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::ObserveUpdate {
            queue: key,
            status: QueueSyscallStatusV1::FailedNoEffect,
        },
    );
    assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Disabled);
}

#[test]
fn hostile_resource_alias_and_mapping_substitution_reject_before_publication() {
    let fixture = fixture();
    let state = QueueLifecycleStateV1::new(domain(1));
    let mut aliased = fixture.plan;
    aliased.resources.control = aliased.resources.ring;
    assert!(matches!(
        state.admit_compute_aql_plan(&fixture.identity, &fixture.memory, aliased),
        Err(QueueTransitionErrorV1::InvalidPlan(
            QueueInvariantViolationV1::ResourceAlias(_)
        ))
    ));

    let mut substituted = fixture.plan;
    substituted.resources.eop.mapping.id = MappingIdV1(9_999);
    substituted.resources.eop.publication.mapping = substituted.resources.eop.mapping;
    assert!(matches!(
        state.admit_compute_aql_plan(&fixture.identity, &fixture.memory, substituted),
        Err(QueueTransitionErrorV1::InvalidPlan(
            QueueInvariantViolationV1::MissingResource(_, ComputeAqlResourceRoleV1::EndOfPipe)
        ))
    ));
    assert!(fixture.memory.publications().is_empty());

    let mut policy_substituted = fixture.plan;
    policy_substituted.resources.eop.expected_kind = MemoryKindV1::Executable;
    assert!(matches!(
        state.admit_compute_aql_plan(&fixture.identity, &fixture.memory, policy_substituted),
        Err(QueueTransitionErrorV1::InvalidPlan(
            QueueInvariantViolationV1::ResourceReleasedEarly(
                _,
                ComputeAqlResourceRoleV1::EndOfPipe
            )
        ))
    ));
}

#[test]
fn retaining_queue_plans_cannot_share_any_mapped_resource() {
    let fixture = fixture();
    let admission = admit(&fixture);
    let (queue, memory) = admission.into_states();
    let mut shared = fixture.plan;
    shared.plan_id = QueuePlanIdV1::from_untrusted_digest(digest(20));
    shared.queue.id = QueueInstanceIdV1(701);
    shared.initial_configuration = QueueConfigurationIdV1::from_untrusted_digest(digest(21));
    for (index, (_, resource)) in shared.resources.ordered().iter().enumerate() {
        let publication_id = MemoryPublicationIdV1(900 + index as u64);
        match index {
            0 => shared.resources.ring.publication.id = publication_id,
            1 => shared.resources.control.publication.id = publication_id,
            2 => shared.resources.eop.publication.id = publication_id,
            3 => shared.resources.context_save.publication.id = publication_id,
            _ => unreachable!(),
        }
        assert_eq!(resource.mapping.allocation.vm, fixture.plan.queue.vm);
    }
    assert!(matches!(
        queue.admit_compute_aql_plan(&fixture.identity, &memory, shared),
        Err(QueueTransitionErrorV1::InvalidPlan(
            QueueInvariantViolationV1::ResourceAlias(_)
        ))
    ));
}

#[test]
fn stale_device_or_vm_history_rejects_queue_use() {
    let fixture = fixture();
    let identity = fixture
        .identity
        .retire_vm_model_only(fixture.vm)
        .unwrap()
        .retire_device_model_only(fixture.device)
        .unwrap();
    assert!(matches!(
        QueueLifecycleStateV1::new(domain(1)).admit_compute_aql_plan(
            &identity,
            &fixture.memory,
            fixture.plan
        ),
        Err(QueueTransitionErrorV1::InvalidPlan(
            QueueInvariantViolationV1::DeviceNotCurrent(_)
                | QueueInvariantViolationV1::VmNotCurrent(_)
        ))
    ));
}

#[test]
fn currentness_loss_quarantines_without_releasing_memory_publications() {
    let fixture = fixture();
    let (queue, memory) = admit(&fixture).into_states();
    let queue = queue
        .quarantine_currentness_loss(fixture.plan.queue)
        .unwrap();
    assert_eq!(queue.queues()[0].phase, ComputeAqlQueuePhaseV1::Ambiguous);
    assert!(
        queue
            .release_resource_publications(&memory, fixture.plan.queue)
            .is_err()
    );
    assert!(
        memory
            .publications()
            .iter()
            .all(|record| record.state == MemoryPublicationStateV1::Live)
    );
}

#[test]
fn currentness_loss_can_quarantine_multiple_queues_and_poison_future_create() {
    let fixture = fixture();
    let (memory, second_resources) =
        append_resource_set(fixture.memory.clone(), &fixture, 1_000, 0x8_0000);
    let (memory, third_resources) = append_resource_set(memory, &fixture, 2_000, 0xc_0000);
    let second_plan = distinct_plan(&fixture, second_resources, 701, 1, 51);
    let third_plan = distinct_plan(&fixture, third_resources, 702, 1, 52);
    let first = QueueLifecycleStateV1::new(domain(1))
        .admit_compute_aql_plan(&fixture.identity, &memory, fixture.plan)
        .unwrap();
    let (queue, memory) = first.into_states();
    let second = queue
        .admit_compute_aql_plan(&fixture.identity, &memory, second_plan)
        .unwrap();
    let (queue, memory) = second.into_states();
    let third = queue
        .admit_compute_aql_plan(&fixture.identity, &memory, third_plan)
        .unwrap();
    let (queue, memory) = third.into_states();

    let queue = queue
        .quarantine_currentness_loss(fixture.plan.queue)
        .unwrap()
        .quarantine_currentness_loss(second_plan.queue)
        .unwrap();
    assert!(queue.queues()[..2].iter().all(|record| {
        record.phase == ComputeAqlQueuePhaseV1::Ambiguous && record.queue_id.is_none()
    }));
    assert!(
        queue.queues()[..2]
            .iter()
            .all(|record| record.phase.retains_resources())
    );
    assert_eq!(
        memory
            .publications()
            .iter()
            .filter(|publication| publication.state == MemoryPublicationStateV1::Live)
            .count(),
        3 * COMPUTE_AQL_RESOURCE_COUNT_V1
    );
    assert!(matches!(
        queue.next(
            &fixture.identity,
            &memory,
            QueueTransitionV1::BeginCreate {
                queue: third_plan.queue,
            },
        ),
        Err(QueueTransitionErrorV1::QueueCreationPoisoned(key))
            if key == third_plan.queue
    ));
}

#[test]
fn illegal_early_destroy_and_configuration_reuse_reject_without_history_growth() {
    let fixture = fixture();
    let (queue, memory) = admit(&fixture).into_states();
    let history_len = queue.history().len();
    assert!(
        queue
            .next(
                &fixture.identity,
                &memory,
                QueueTransitionV1::BeginDestroy {
                    queue: fixture.plan.queue,
                },
            )
            .is_err()
    );
    assert_eq!(queue.history().len(), history_len);

    let queue = create_active(queue, &fixture, &memory);
    assert!(matches!(
        queue.next(
            &fixture.identity,
            &memory,
            QueueTransitionV1::BeginUpdate {
                queue: fixture.plan.queue,
                configuration: fixture.plan.initial_configuration,
            },
        ),
        Err(QueueTransitionErrorV1::InvalidConfiguration(_))
    ));
}

#[test]
fn generic_memory_release_cannot_bypass_active_or_ambiguous_queue_ownership() {
    for ambiguous in [false, true] {
        let fixture = fixture();
        let (queue, memory) = admit(&fixture).into_states();
        assert!(memory.publications().iter().all(|publication| {
            publication.owner == MemoryPublicationOwnerV1::ComputeAqlQueue(fixture.plan.queue)
        }));
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::BeginCreate {
                queue: fixture.plan.queue,
            },
        );
        let status = if ambiguous {
            QueueSyscallStatusV1::Indeterminate
        } else {
            QueueSyscallStatusV1::Succeeded
        };
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::ObserveCreate {
                queue: fixture.plan.queue,
                observation: QueueCreateObservationV1 {
                    status,
                    queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                        UntrustedQueueIdObservationV1(17),
                    ),
                },
            },
        );
        assert_eq!(
            queue.queues()[0].phase,
            if ambiguous {
                ComputeAqlQueuePhaseV1::Ambiguous
            } else {
                ComputeAqlQueuePhaseV1::Active
            }
        );
        for (_, resource) in fixture.plan.resources.ordered() {
            assert!(matches!(
                memory.next(MemoryTransitionV1::ReleasePublication {
                    key: resource.publication,
                }),
                Err(MemoryTransitionErrorV1::ResourceInUse(
                    MemoryRecordRefV1::Publication(_)
                ))
            ));
            assert!(matches!(
                memory.next(MemoryTransitionV1::BeginUnmap {
                    key: resource.mapping,
                }),
                Err(MemoryTransitionErrorV1::ResourceInUse(
                    MemoryRecordRefV1::Mapping(_)
                ))
            ));
        }
    }
}

#[test]
fn ambiguous_known_id_collides_and_unknown_id_poisons_later_create() {
    for known_id in [true, false] {
        let fixture = fixture();
        let (memory, second_resources) =
            append_resource_set(fixture.memory.clone(), &fixture, 1_000, 0x8_0000);
        let second_plan = distinct_plan(&fixture, second_resources, 701, 1, 30);
        let first = QueueLifecycleStateV1::new(domain(1))
            .admit_compute_aql_plan(&fixture.identity, &memory, fixture.plan)
            .unwrap();
        let (queue, memory) = first.into_states();
        let second = queue
            .admit_compute_aql_plan(&fixture.identity, &memory, second_plan)
            .unwrap();
        let (queue, memory) = second.into_states();
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::BeginCreate {
                queue: fixture.plan.queue,
            },
        );
        let queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::ObserveCreate {
                queue: fixture.plan.queue,
                observation: QueueCreateObservationV1 {
                    status: QueueSyscallStatusV1::Indeterminate,
                    queue_id_field: if known_id {
                        CreateQueueIdFieldObservationV1::Returned(UntrustedQueueIdObservationV1(23))
                    } else {
                        CreateQueueIdFieldObservationV1::SentinelUnchanged
                    },
                },
            },
        );
        if known_id {
            let queue = advance(
                queue,
                &fixture,
                &memory,
                QueueTransitionV1::BeginCreate {
                    queue: second_plan.queue,
                },
            );
            let queue = advance(
                queue,
                &fixture,
                &memory,
                QueueTransitionV1::ObserveCreate {
                    queue: second_plan.queue,
                    observation: QueueCreateObservationV1 {
                        status: QueueSyscallStatusV1::Succeeded,
                        queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                            UntrustedQueueIdObservationV1(23),
                        ),
                    },
                },
            );
            assert_eq!(queue.queues()[1].phase, ComputeAqlQueuePhaseV1::Ambiguous);
            assert_eq!(queue.queues()[1].queue_id, None);
        } else {
            assert!(matches!(
                queue.next(
                    &fixture.identity,
                    &memory,
                    QueueTransitionV1::BeginCreate {
                        queue: second_plan.queue,
                    },
                ),
                Err(QueueTransitionErrorV1::QueueCreationPoisoned(key))
                    if key == second_plan.queue
            ));
        }
    }
}

#[test]
fn a_second_create_cannot_begin_while_the_first_identity_is_unresolved() {
    let fixture = fixture();
    let (memory, second_resources) =
        append_resource_set(fixture.memory.clone(), &fixture, 1_000, 0x8_0000);
    let second_plan = distinct_plan(&fixture, second_resources, 701, 1, 45);
    let first = QueueLifecycleStateV1::new(domain(1))
        .admit_compute_aql_plan(&fixture.identity, &memory, fixture.plan)
        .unwrap();
    let (queue, memory) = first.into_states();
    let second = queue
        .admit_compute_aql_plan(&fixture.identity, &memory, second_plan)
        .unwrap();
    let (queue, memory) = second.into_states();
    let queue = advance(
        queue,
        &fixture,
        &memory,
        QueueTransitionV1::BeginCreate {
            queue: fixture.plan.queue,
        },
    );
    assert!(matches!(
        queue.next(
            &fixture.identity,
            &memory,
            QueueTransitionV1::BeginCreate {
                queue: second_plan.queue,
            },
        ),
        Err(QueueTransitionErrorV1::QueueCreationPoisoned(key))
            if key == second_plan.queue
    ));
}

#[test]
fn two_disjoint_queues_can_be_active_with_distinct_returned_ids() {
    let fixture = fixture();
    let (memory, second_resources) =
        append_resource_set(fixture.memory.clone(), &fixture, 1_000, 0x8_0000);
    let second_plan = distinct_plan(&fixture, second_resources, 701, 1, 40);
    let first = QueueLifecycleStateV1::new(domain(1))
        .admit_compute_aql_plan(&fixture.identity, &memory, fixture.plan)
        .unwrap();
    let (queue, memory) = first.into_states();
    let second = queue
        .admit_compute_aql_plan(&fixture.identity, &memory, second_plan)
        .unwrap();
    let (mut queue, memory) = second.into_states();
    for (key, queue_id) in [(fixture.plan.queue, 0), (second_plan.queue, 1)] {
        queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::BeginCreate { queue: key },
        );
        queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::ObserveCreate {
                queue: key,
                observation: QueueCreateObservationV1 {
                    status: QueueSyscallStatusV1::Succeeded,
                    queue_id_field: CreateQueueIdFieldObservationV1::Returned(
                        UntrustedQueueIdObservationV1(queue_id),
                    ),
                },
            },
        );
    }
    assert!(
        queue
            .queues()
            .iter()
            .all(|record| record.phase == ComputeAqlQueuePhaseV1::Active)
    );
}

#[test]
fn queue_generations_are_contiguous_and_queue_capacity_is_fail_closed() {
    let fixture = fixture();
    let mut queue = QueueLifecycleStateV1::new(domain(1));
    let mut memory = fixture.memory.clone();
    for generation in 1..=MAX_COMPUTE_AQL_QUEUES_V1 as u64 {
        let mut plan = fixture.plan;
        plan.queue.generation = QueueGenerationV1(generation);
        plan.plan_id = QueuePlanIdV1::from_untrusted_digest(digest(50 + generation as u8));
        plan.initial_configuration =
            QueueConfigurationIdV1::from_untrusted_digest(digest(80 + generation as u8));
        for (_, resource) in plan.resources.ordered().iter() {
            let id = resource.publication.id.0 + generation * 100;
            if resource.mapping == plan.resources.ring.mapping {
                plan.resources.ring.publication.id = MemoryPublicationIdV1(id);
            } else if resource.mapping == plan.resources.control.mapping {
                plan.resources.control.publication.id = MemoryPublicationIdV1(id);
            } else if resource.mapping == plan.resources.eop.mapping {
                plan.resources.eop.publication.id = MemoryPublicationIdV1(id);
            } else {
                plan.resources.context_save.publication.id = MemoryPublicationIdV1(id);
            }
        }
        let result = queue
            .admit_compute_aql_plan(&fixture.identity, &memory, plan)
            .unwrap();
        (queue, memory) = result.into_states();
        queue = advance(
            queue,
            &fixture,
            &memory,
            QueueTransitionV1::CancelPlan { queue: plan.queue },
        );
        memory = queue
            .release_resource_publications(&memory, plan.queue)
            .unwrap();
    }
    let mut over_capacity = fixture.plan;
    over_capacity.queue.generation = QueueGenerationV1(MAX_COMPUTE_AQL_QUEUES_V1 as u64 + 1);
    assert!(matches!(
        queue.admit_compute_aql_plan(&fixture.identity, &memory, over_capacity),
        Err(QueueTransitionErrorV1::CapacityExceeded {
            kind: QueueRecordKindV1::Queue,
            maximum: MAX_COMPUTE_AQL_QUEUES_V1,
        })
    ));
}

#[test]
fn four_publication_plan_is_failure_atomic_when_a_later_stage_hits_capacity() {
    let fixture = fixture();
    let memory = fixture.memory.with_generic_publications_for_test(
        fixture.plan.resources.ring.mapping,
        MAX_MEMORY_PUBLICATIONS_V1 - 2,
    );
    let queue = QueueLifecycleStateV1::new(domain(1));
    let original_queue = queue.clone();
    let original_memory = memory.clone();
    assert!(matches!(
        queue.admit_compute_aql_plan(&fixture.identity, &memory, fixture.plan),
        Err(QueueTransitionErrorV1::Memory(
            MemoryTransitionErrorV1::CapacityExceeded {
                kind: MemoryRecordKindV1::Publication,
                maximum: MAX_MEMORY_PUBLICATIONS_V1,
            }
        ))
    ));
    assert_eq!(queue, original_queue);
    assert_eq!(memory, original_memory);
    assert_eq!(memory.publications().len(), MAX_MEMORY_PUBLICATIONS_V1 - 2);
}
