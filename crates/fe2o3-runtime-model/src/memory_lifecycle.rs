//! Bounded, syscall-free VM and GPU-memory lifecycle model.

use alloc::vec::Vec;

use crate::*;

pub const MEMORY_LIFECYCLE_SCHEMA_VERSION_V1: u16 = 1;
pub const MEMORY_PAGE_BYTES_V1: u64 = 4_096;
pub const MAX_MEMORY_VMS_V1: usize = 64;
pub const MAX_VA_RESERVATIONS_V1: usize = 512;
pub const MAX_MEMORY_ALLOCATIONS_V1: usize = 512;
pub const MAX_MEMORY_MAPPINGS_V1: usize = 1_024;
pub const MAX_MEMORY_PUBLICATIONS_V1: usize = 2_048;
pub const MAX_MEMORY_MAPPING_DEVICES_V1: usize = 16;

/// Untrusted process-local VM handle observation. It is not durable identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct UntrustedVmHandleObservationV1(pub u64);

/// Untrusted KFD allocation-handle observation. It is bound to an allocation
/// generation but does not grant allocation or file-descriptor authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct UntrustedAllocationHandleObservationV1(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VaReservationKeyV1 {
    pub vm: VmKeyV1,
    pub id: VaReservationIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryAllocationKeyV1 {
    pub vm: VmKeyV1,
    pub id: AllocationIdV1,
    pub generation: AllocationGenerationV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryMappingKeyV1 {
    pub allocation: MemoryAllocationKeyV1,
    pub id: MappingIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryPublicationKeyV1 {
    pub mapping: MemoryMappingKeyV1,
    pub id: MemoryPublicationIdV1,
}

/// Half-open GPU virtual-address interval `[base, base + byte_len)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuVaRangeV1 {
    pub base: u64,
    pub byte_len: u64,
}

impl GpuVaRangeV1 {
    pub const fn checked_end(self) -> Option<u64> {
        self.base.checked_add(self.byte_len)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKindV1 {
    HostVisibleCoherent,
    QueueStorage,
    Kernarg,
    Executable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryCoherenceV1 {
    HostCoherent,
    ExplicitVisibility,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryAllocationSpecV1 {
    pub byte_len: u64,
    pub alignment: u64,
    pub kind: MemoryKindV1,
    pub coherence: MemoryCoherenceV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryVmStateV1 {
    Active,
    Retired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaReservationStateV1 {
    Reserved,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAllocationStateV1 {
    Live,
    Released,
}

/// State of one exact map/unmap device-list operation.
///
/// The retained indices describe the conservative device subrange that may
/// still be mapped. Map progress establishes `[0, mapped_end)`. Unmap progress
/// reports an absolute cumulative prefix boundary and assigns `mapped_start`
/// to that boundary. `Ambiguous` never permits release, regardless of the
/// retained range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryMappingStateV1 {
    MapPending,
    MapFailed,
    Mapped,
    UnmapPending,
    UnmapFailed,
    Unmapped,
    Ambiguous,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPublicationStateV1 {
    Live,
    Released,
}

/// Structural owner of a mapping-retention publication.
///
/// Queue-owned publications can only be minted and released by the joint queue
/// lifecycle transition. Public generic memory transitions cannot discharge
/// them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPublicationOwnerV1 {
    Generic,
    ComputeAqlQueue(QueueKeyV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartialOperationStatusV1 {
    Succeeded,
    Failed,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartialProgressObservationV1 {
    pub n_success: usize,
    pub status: PartialOperationStatusV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryVmRecordV1 {
    pub admission: ModelVmAdmissionV1,
    pub mapping_devices: Vec<ModelDeviceAdmissionV1>,
    pub handle: UntrustedVmHandleObservationV1,
    pub aperture: GpuVaRangeV1,
    pub state: MemoryVmStateV1,
}

impl MemoryVmRecordV1 {
    pub fn mapping_device_keys(&self) -> impl ExactSizeIterator<Item = DeviceKeyV1> + '_ {
        self.mapping_devices
            .iter()
            .map(|admission| admission.model_key())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VaReservationRecordV1 {
    pub key: VaReservationKeyV1,
    pub range: GpuVaRangeV1,
    pub alignment: u64,
    pub state: VaReservationStateV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryAllocationRecordV1 {
    pub key: MemoryAllocationKeyV1,
    pub reservation: VaReservationKeyV1,
    pub handle: UntrustedAllocationHandleObservationV1,
    pub spec: MemoryAllocationSpecV1,
    pub state: MemoryAllocationStateV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryMappingRecordV1 {
    pub key: MemoryMappingKeyV1,
    pub target_devices: Vec<DeviceKeyV1>,
    pub access: MemoryAccessV1,
    pub mapped_start: usize,
    pub mapped_end: usize,
    pub state: MemoryMappingStateV1,
}

impl MemoryMappingRecordV1 {
    /// Conservative device subrange retained against unmap/free.
    pub fn retained_device_superset(&self) -> &[DeviceKeyV1] {
        &self.target_devices[self.mapped_start..self.mapped_end]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPublicationRecordV1 {
    pub key: MemoryPublicationKeyV1,
    pub owner: MemoryPublicationOwnerV1,
    pub state: MemoryPublicationStateV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryRecordKindV1 {
    Vm,
    VaReservation,
    Allocation,
    Mapping,
    Publication,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryRecordRefV1 {
    Vm(VmKeyV1),
    VaReservation(VaReservationKeyV1),
    Allocation(MemoryAllocationKeyV1),
    Mapping(MemoryMappingKeyV1),
    Publication(MemoryPublicationKeyV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryInvariantViolationV1 {
    CapacityExceeded(MemoryRecordKindV1),
    DomainMismatch(MemoryRecordRefV1),
    Duplicate(MemoryRecordRefV1),
    MissingParent(MemoryRecordRefV1),
    BindingMismatch(MemoryRecordRefV1),
    InvalidIdentity(MemoryRecordRefV1),
    InvalidRange(MemoryRecordRefV1),
    InvalidAlignment(MemoryRecordRefV1),
    DeviceSetMismatch(MemoryRecordRefV1),
    HandleCollision(MemoryRecordRefV1),
    StaleGeneration(MemoryAllocationKeyV1),
    AddressOverlap(VaReservationKeyV1, VaReservationKeyV1),
    InvalidState(MemoryRecordRefV1),
    EarlyRelease(MemoryRecordRefV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryTransitionErrorV1 {
    SourceInvariant(MemoryInvariantViolationV1),
    NextInvariant(MemoryInvariantViolationV1),
    CapacityExceeded {
        kind: MemoryRecordKindV1,
        maximum: usize,
    },
    NotFound(MemoryRecordRefV1),
    AlreadyExists(MemoryRecordRefV1),
    ObservationDomainMismatch,
    InvalidIdentity(MemoryRecordRefV1),
    InvalidRange(MemoryRecordRefV1),
    InvalidAlignment(MemoryRecordRefV1),
    BindingMismatch(MemoryRecordRefV1),
    DeviceSetMismatch(MemoryRecordRefV1),
    HandleCollision(MemoryRecordRefV1),
    StaleGeneration(MemoryAllocationKeyV1),
    AddressConflict(VaReservationKeyV1),
    IllegalState(MemoryRecordRefV1),
    ResourceInUse(MemoryRecordRefV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryTransitionV1 {
    AcquireVm {
        admission: ModelVmAdmissionV1,
        mapping_devices: Vec<ModelDeviceAdmissionV1>,
        handle: UntrustedVmHandleObservationV1,
        aperture: GpuVaRangeV1,
    },
    RetireVm {
        key: VmKeyV1,
    },
    ReserveVa {
        key: VaReservationKeyV1,
        range: GpuVaRangeV1,
        alignment: u64,
    },
    ReleaseVaReservation {
        key: VaReservationKeyV1,
    },
    Allocate {
        key: MemoryAllocationKeyV1,
        reservation: VaReservationKeyV1,
        handle: UntrustedAllocationHandleObservationV1,
        spec: MemoryAllocationSpecV1,
    },
    ReleaseAllocation {
        key: MemoryAllocationKeyV1,
    },
    BeginMap {
        key: MemoryMappingKeyV1,
        target_devices: Vec<DeviceKeyV1>,
        access: MemoryAccessV1,
    },
    ObserveMap {
        key: MemoryMappingKeyV1,
        progress: PartialProgressObservationV1,
    },
    BeginUnmap {
        key: MemoryMappingKeyV1,
    },
    ObserveUnmap {
        key: MemoryMappingKeyV1,
        progress: PartialProgressObservationV1,
    },
    ReleaseMapping {
        key: MemoryMappingKeyV1,
    },
    PublishMapping {
        key: MemoryPublicationKeyV1,
    },
    ReleasePublication {
        key: MemoryPublicationKeyV1,
    },
}

/// Append-only process-domain memory history.
///
/// All inputs and receipts remain model-only. External adapters must translate
/// every possibly side-effecting malformed/unknown result to `Indeterminate`;
/// rejecting a pure transition alone is not rollback evidence for a syscall.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLifecycleStateV1 {
    domain_id: DeviceObservationDomainIdV1,
    vms: Vec<MemoryVmRecordV1>,
    reservations: Vec<VaReservationRecordV1>,
    allocations: Vec<MemoryAllocationRecordV1>,
    mappings: Vec<MemoryMappingRecordV1>,
    publications: Vec<MemoryPublicationRecordV1>,
}

impl MemoryLifecycleStateV1 {
    pub const fn new(domain_id: DeviceObservationDomainIdV1) -> Self {
        Self {
            domain_id,
            vms: Vec::new(),
            reservations: Vec::new(),
            allocations: Vec::new(),
            mappings: Vec::new(),
            publications: Vec::new(),
        }
    }

    pub const fn authority_domain(&self) -> AuthorityDomainV1 {
        AuthorityDomainV1::ModelOnly
    }

    pub const fn domain_id(&self) -> DeviceObservationDomainIdV1 {
        self.domain_id
    }

    pub fn vms(&self) -> &[MemoryVmRecordV1] {
        &self.vms
    }

    pub fn reservations(&self) -> &[VaReservationRecordV1] {
        &self.reservations
    }

    pub fn allocations(&self) -> &[MemoryAllocationRecordV1] {
        &self.allocations
    }

    pub fn mappings(&self) -> &[MemoryMappingRecordV1] {
        &self.mappings
    }

    pub fn publications(&self) -> &[MemoryPublicationRecordV1] {
        &self.publications
    }

    #[cfg(test)]
    pub(crate) fn with_generic_publications_for_test(
        &self,
        mapping: MemoryMappingKeyV1,
        count: usize,
    ) -> Self {
        let mut next = self.clone();
        for offset in 0..count {
            next.publications.push(MemoryPublicationRecordV1 {
                key: MemoryPublicationKeyV1 {
                    mapping,
                    id: MemoryPublicationIdV1(10_000 + offset as u64),
                },
                owner: MemoryPublicationOwnerV1::Generic,
                state: MemoryPublicationStateV1::Live,
            });
        }
        assert!(next.validate_global_invariants().is_ok());
        next
    }

    pub fn next(&self, transition: MemoryTransitionV1) -> Result<Self, MemoryTransitionErrorV1> {
        self.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::SourceInvariant)?;
        let mut next = self.clone();
        next.apply(transition)?;
        next.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::NextInvariant)?;
        Ok(next)
    }

    pub(crate) fn publish_compute_aql_queue_mapping(
        &self,
        key: MemoryPublicationKeyV1,
        queue: QueueKeyV1,
    ) -> Result<Self, MemoryTransitionErrorV1> {
        self.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::SourceInvariant)?;
        let mut next = self.clone();
        next.publish_mapping(key, MemoryPublicationOwnerV1::ComputeAqlQueue(queue))?;
        next.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::NextInvariant)?;
        Ok(next)
    }

    pub(crate) fn release_compute_aql_queue_publication(
        &self,
        key: MemoryPublicationKeyV1,
        queue: QueueKeyV1,
    ) -> Result<Self, MemoryTransitionErrorV1> {
        self.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::SourceInvariant)?;
        let mut next = self.clone();
        next.release_queue_publication(key, queue)?;
        next.validate_global_invariants()
            .map_err(MemoryTransitionErrorV1::NextInvariant)?;
        Ok(next)
    }

    pub fn validate_global_invariants(&self) -> Result<(), MemoryInvariantViolationV1> {
        self.validate_capacities()?;
        self.validate_vms()?;
        self.validate_reservations()?;
        self.validate_allocations()?;
        self.validate_mappings()?;
        self.validate_publications()?;
        self.validate_release_order()?;
        Ok(())
    }

    fn apply(&mut self, transition: MemoryTransitionV1) -> Result<(), MemoryTransitionErrorV1> {
        match transition {
            MemoryTransitionV1::AcquireVm {
                admission,
                mapping_devices,
                handle,
                aperture,
            } => self.acquire_vm(admission, mapping_devices, handle, aperture),
            MemoryTransitionV1::RetireVm { key } => self.retire_vm(key),
            MemoryTransitionV1::ReserveVa {
                key,
                range,
                alignment,
            } => self.reserve_va(key, range, alignment),
            MemoryTransitionV1::ReleaseVaReservation { key } => self.release_reservation(key),
            MemoryTransitionV1::Allocate {
                key,
                reservation,
                handle,
                spec,
            } => self.allocate(key, reservation, handle, spec),
            MemoryTransitionV1::ReleaseAllocation { key } => self.release_allocation(key),
            MemoryTransitionV1::BeginMap {
                key,
                target_devices,
                access,
            } => self.begin_map(key, target_devices, access),
            MemoryTransitionV1::ObserveMap { key, progress } => self.observe_map(key, progress),
            MemoryTransitionV1::BeginUnmap { key } => self.begin_unmap(key),
            MemoryTransitionV1::ObserveUnmap { key, progress } => self.observe_unmap(key, progress),
            MemoryTransitionV1::ReleaseMapping { key } => self.release_mapping(key),
            MemoryTransitionV1::PublishMapping { key } => {
                self.publish_mapping(key, MemoryPublicationOwnerV1::Generic)
            }
            MemoryTransitionV1::ReleasePublication { key } => self.release_publication(key),
        }
    }

    fn acquire_vm(
        &mut self,
        admission: ModelVmAdmissionV1,
        mapping_devices: Vec<ModelDeviceAdmissionV1>,
        handle: UntrustedVmHandleObservationV1,
        aperture: GpuVaRangeV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        ensure_memory_room(self.vms.len(), MAX_MEMORY_VMS_V1, MemoryRecordKindV1::Vm)?;
        let key = admission.model_key();
        let reference = MemoryRecordRefV1::Vm(key);
        if admission.domain_id() != self.domain_id
            || mapping_devices
                .iter()
                .any(|device| device.domain_id() != self.domain_id)
        {
            return Err(MemoryTransitionErrorV1::ObservationDomainMismatch);
        }
        if key.id.0 == 0 || handle.0 == 0 {
            return Err(MemoryTransitionErrorV1::InvalidIdentity(reference));
        }
        if self
            .vms
            .iter()
            .any(|record| record.admission.model_key() == key)
        {
            return Err(MemoryTransitionErrorV1::AlreadyExists(reference));
        }
        if self
            .vms
            .iter()
            .any(|record| record.state == MemoryVmStateV1::Active && record.handle == handle)
        {
            return Err(MemoryTransitionErrorV1::HandleCollision(reference));
        }
        if !valid_mapping_admissions(&mapping_devices, key.device) {
            return Err(MemoryTransitionErrorV1::DeviceSetMismatch(reference));
        }
        if !valid_page_range(aperture) {
            return Err(MemoryTransitionErrorV1::InvalidRange(reference));
        }
        self.vms.push(MemoryVmRecordV1 {
            admission,
            mapping_devices,
            handle,
            aperture,
            state: MemoryVmStateV1::Active,
        });
        Ok(())
    }

    fn retire_vm(&mut self, key: VmKeyV1) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Vm(key);
        if self.vm(key)?.state != MemoryVmStateV1::Active {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        if self
            .reservations
            .iter()
            .any(|record| record.key.vm == key && record.state != VaReservationStateV1::Released)
            || self.allocations.iter().any(|record| {
                record.key.vm == key && record.state != MemoryAllocationStateV1::Released
            })
            || self.mappings.iter().any(|record| {
                record.key.allocation.vm == key && record.state != MemoryMappingStateV1::Released
            })
            || self.publications.iter().any(|record| {
                record.key.mapping.allocation.vm == key
                    && record.state != MemoryPublicationStateV1::Released
            })
        {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        self.vm_mut(key)?.state = MemoryVmStateV1::Retired;
        Ok(())
    }

    fn reserve_va(
        &mut self,
        key: VaReservationKeyV1,
        range: GpuVaRangeV1,
        alignment: u64,
    ) -> Result<(), MemoryTransitionErrorV1> {
        ensure_memory_room(
            self.reservations.len(),
            MAX_VA_RESERVATIONS_V1,
            MemoryRecordKindV1::VaReservation,
        )?;
        let reference = MemoryRecordRefV1::VaReservation(key);
        self.require_vm_active(key.vm)?;
        if key.id.0 == 0 {
            return Err(MemoryTransitionErrorV1::InvalidIdentity(reference));
        }
        if self.reservations.iter().any(|record| record.key == key) {
            return Err(MemoryTransitionErrorV1::AlreadyExists(reference));
        }
        if !valid_alignment(alignment) || !range.base.is_multiple_of(alignment) {
            return Err(MemoryTransitionErrorV1::InvalidAlignment(reference));
        }
        if !valid_page_range(range) || !range_within(range, self.vm(key.vm)?.aperture) {
            return Err(MemoryTransitionErrorV1::InvalidRange(reference));
        }
        if self.reservations.iter().any(|record| {
            record.key.vm == key.vm
                && record.state == VaReservationStateV1::Reserved
                && ranges_overlap(record.range, range)
        }) {
            return Err(MemoryTransitionErrorV1::AddressConflict(key));
        }
        self.reservations.push(VaReservationRecordV1 {
            key,
            range,
            alignment,
            state: VaReservationStateV1::Reserved,
        });
        Ok(())
    }

    fn release_reservation(
        &mut self,
        key: VaReservationKeyV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::VaReservation(key);
        if self.reservation(key)?.state != VaReservationStateV1::Reserved {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        if self.allocations.iter().any(|record| {
            record.reservation == key && record.state != MemoryAllocationStateV1::Released
        }) {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        self.reservation_mut(key)?.state = VaReservationStateV1::Released;
        Ok(())
    }

    fn allocate(
        &mut self,
        key: MemoryAllocationKeyV1,
        reservation: VaReservationKeyV1,
        handle: UntrustedAllocationHandleObservationV1,
        spec: MemoryAllocationSpecV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        ensure_memory_room(
            self.allocations.len(),
            MAX_MEMORY_ALLOCATIONS_V1,
            MemoryRecordKindV1::Allocation,
        )?;
        let reference = MemoryRecordRefV1::Allocation(key);
        self.require_vm_active(key.vm)?;
        if key.vm != reservation.vm {
            return Err(MemoryTransitionErrorV1::BindingMismatch(reference));
        }
        if key.id.0 == 0 || key.generation.0 == 0 || handle.0 == 0 {
            return Err(MemoryTransitionErrorV1::InvalidIdentity(reference));
        }
        if self.allocations.iter().any(|record| record.key == key) {
            return Err(MemoryTransitionErrorV1::AlreadyExists(reference));
        }
        let reservation_record = self.reservation(reservation)?;
        if reservation_record.state != VaReservationStateV1::Reserved {
            return Err(MemoryTransitionErrorV1::IllegalState(
                MemoryRecordRefV1::VaReservation(reservation),
            ));
        }
        if !valid_alignment(spec.alignment)
            || !reservation_record.range.base.is_multiple_of(spec.alignment)
            || spec.alignment > reservation_record.alignment
        {
            return Err(MemoryTransitionErrorV1::InvalidAlignment(reference));
        }
        if !valid_allocation_spec(spec, *reservation_record) {
            return Err(MemoryTransitionErrorV1::InvalidRange(reference));
        }
        for old in self
            .allocations
            .iter()
            .filter(|record| record.key.vm == key.vm && record.key.id == key.id)
        {
            if old.state != MemoryAllocationStateV1::Released {
                return Err(MemoryTransitionErrorV1::ResourceInUse(
                    MemoryRecordRefV1::Allocation(old.key),
                ));
            }
            if old.key.generation >= key.generation {
                return Err(MemoryTransitionErrorV1::StaleGeneration(key));
            }
        }
        if self.allocations.iter().any(|record| {
            record.state == MemoryAllocationStateV1::Live
                && (record.reservation == reservation || record.handle == handle)
        }) {
            return Err(MemoryTransitionErrorV1::HandleCollision(reference));
        }
        self.allocations.push(MemoryAllocationRecordV1 {
            key,
            reservation,
            handle,
            spec,
            state: MemoryAllocationStateV1::Live,
        });
        Ok(())
    }

    fn release_allocation(
        &mut self,
        key: MemoryAllocationKeyV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Allocation(key);
        if self.allocation(key)?.state != MemoryAllocationStateV1::Live {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        if self.mappings.iter().any(|record| {
            record.key.allocation == key && record.state != MemoryMappingStateV1::Released
        }) || self.publications.iter().any(|record| {
            record.key.mapping.allocation == key
                && record.state != MemoryPublicationStateV1::Released
        }) {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        self.allocation_mut(key)?.state = MemoryAllocationStateV1::Released;
        Ok(())
    }

    fn begin_map(
        &mut self,
        key: MemoryMappingKeyV1,
        target_devices: Vec<DeviceKeyV1>,
        access: MemoryAccessV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        ensure_memory_room(
            self.mappings.len(),
            MAX_MEMORY_MAPPINGS_V1,
            MemoryRecordKindV1::Mapping,
        )?;
        let reference = MemoryRecordRefV1::Mapping(key);
        self.require_vm_active(key.allocation.vm)?;
        if key.id.0 == 0 {
            return Err(MemoryTransitionErrorV1::InvalidIdentity(reference));
        }
        if self.mappings.iter().any(|record| record.key == key) {
            return Err(MemoryTransitionErrorV1::AlreadyExists(reference));
        }
        if self.allocation(key.allocation)?.state != MemoryAllocationStateV1::Live {
            return Err(MemoryTransitionErrorV1::IllegalState(
                MemoryRecordRefV1::Allocation(key.allocation),
            ));
        }
        let vm = self.vm(key.allocation.vm)?;
        if !device_keys_equal_admissions(&target_devices, &vm.mapping_devices) {
            return Err(MemoryTransitionErrorV1::DeviceSetMismatch(reference));
        }
        self.mappings.push(MemoryMappingRecordV1 {
            key,
            target_devices,
            access,
            mapped_start: 0,
            mapped_end: 0,
            state: MemoryMappingStateV1::MapPending,
        });
        Ok(())
    }

    fn observe_map(
        &mut self,
        key: MemoryMappingKeyV1,
        progress: PartialProgressObservationV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Mapping(key);
        let mapping = self.mapping_mut(key)?;
        if mapping.state != MemoryMappingStateV1::MapPending {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        let count = mapping.target_devices.len();
        match progress.status {
            PartialOperationStatusV1::Succeeded if progress.n_success == count => {
                mapping.mapped_end = count;
                mapping.state = MemoryMappingStateV1::Mapped;
            }
            PartialOperationStatusV1::Failed if progress.n_success <= count => {
                mapping.mapped_end = progress.n_success;
                mapping.state = MemoryMappingStateV1::MapFailed;
            }
            PartialOperationStatusV1::Succeeded
            | PartialOperationStatusV1::Failed
            | PartialOperationStatusV1::Indeterminate => {
                mapping.mapped_start = 0;
                mapping.mapped_end = count;
                mapping.state = MemoryMappingStateV1::Ambiguous;
            }
        }
        Ok(())
    }

    fn begin_unmap(&mut self, key: MemoryMappingKeyV1) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Mapping(key);
        if self.publications.iter().any(|publication| {
            publication.key.mapping == key && publication.state == MemoryPublicationStateV1::Live
        }) {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        let mapping = self.mapping_mut(key)?;
        if !matches!(
            mapping.state,
            MemoryMappingStateV1::Mapped
                | MemoryMappingStateV1::MapFailed
                | MemoryMappingStateV1::UnmapFailed
        ) || mapping.mapped_start == mapping.mapped_end
        {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        mapping.state = MemoryMappingStateV1::UnmapPending;
        Ok(())
    }

    fn observe_unmap(
        &mut self,
        key: MemoryMappingKeyV1,
        progress: PartialProgressObservationV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Mapping(key);
        let mapping = self.mapping_mut(key)?;
        if mapping.state != MemoryMappingStateV1::UnmapPending {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        let previous_start = mapping.mapped_start;
        let mapped_end = mapping.mapped_end;
        match progress.status {
            PartialOperationStatusV1::Succeeded if progress.n_success == mapped_end => {
                mapping.mapped_start = mapped_end;
                mapping.state = MemoryMappingStateV1::Unmapped;
            }
            PartialOperationStatusV1::Failed
                if previous_start <= progress.n_success && progress.n_success < mapped_end =>
            {
                mapping.mapped_start = progress.n_success;
                mapping.state = MemoryMappingStateV1::UnmapFailed;
            }
            PartialOperationStatusV1::Succeeded
            | PartialOperationStatusV1::Failed
            | PartialOperationStatusV1::Indeterminate => {
                mapping.state = MemoryMappingStateV1::Ambiguous;
            }
        }
        Ok(())
    }

    fn release_mapping(&mut self, key: MemoryMappingKeyV1) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Mapping(key);
        if self.publications.iter().any(|publication| {
            publication.key.mapping == key && publication.state == MemoryPublicationStateV1::Live
        }) {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        let mapping = self.mapping_mut(key)?;
        if !matches!(
            mapping.state,
            MemoryMappingStateV1::MapFailed
                | MemoryMappingStateV1::UnmapFailed
                | MemoryMappingStateV1::Unmapped
        ) || mapping.mapped_start != mapping.mapped_end
        {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        mapping.state = MemoryMappingStateV1::Released;
        Ok(())
    }

    fn publish_mapping(
        &mut self,
        key: MemoryPublicationKeyV1,
        owner: MemoryPublicationOwnerV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        ensure_memory_room(
            self.publications.len(),
            MAX_MEMORY_PUBLICATIONS_V1,
            MemoryRecordKindV1::Publication,
        )?;
        let reference = MemoryRecordRefV1::Publication(key);
        if key.id.0 == 0 {
            return Err(MemoryTransitionErrorV1::InvalidIdentity(reference));
        }
        if self.publications.iter().any(|record| record.key == key) {
            return Err(MemoryTransitionErrorV1::AlreadyExists(reference));
        }
        if self.mapping(key.mapping)?.state != MemoryMappingStateV1::Mapped {
            return Err(MemoryTransitionErrorV1::IllegalState(
                MemoryRecordRefV1::Mapping(key.mapping),
            ));
        }
        if let MemoryPublicationOwnerV1::ComputeAqlQueue(queue) = owner
            && (queue.vm != key.mapping.allocation.vm || queue.id.0 == 0 || queue.generation.0 == 0)
        {
            return Err(MemoryTransitionErrorV1::BindingMismatch(reference));
        }
        self.publications.push(MemoryPublicationRecordV1 {
            key,
            owner,
            state: MemoryPublicationStateV1::Live,
        });
        Ok(())
    }

    fn release_publication(
        &mut self,
        key: MemoryPublicationKeyV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Publication(key);
        let publication = self.publication(key)?;
        if publication.owner != MemoryPublicationOwnerV1::Generic {
            return Err(MemoryTransitionErrorV1::ResourceInUse(reference));
        }
        let publication = self.publication_mut(key)?;
        if publication.state != MemoryPublicationStateV1::Live {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        publication.state = MemoryPublicationStateV1::Released;
        Ok(())
    }

    fn release_queue_publication(
        &mut self,
        key: MemoryPublicationKeyV1,
        queue: QueueKeyV1,
    ) -> Result<(), MemoryTransitionErrorV1> {
        let reference = MemoryRecordRefV1::Publication(key);
        let publication = self.publication(key)?;
        if publication.owner != MemoryPublicationOwnerV1::ComputeAqlQueue(queue) {
            return Err(MemoryTransitionErrorV1::BindingMismatch(reference));
        }
        if publication.state != MemoryPublicationStateV1::Live {
            return Err(MemoryTransitionErrorV1::IllegalState(reference));
        }
        self.publication_mut(key)?.state = MemoryPublicationStateV1::Released;
        Ok(())
    }

    fn validate_capacities(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (actual, maximum, kind) in [
            (self.vms.len(), MAX_MEMORY_VMS_V1, MemoryRecordKindV1::Vm),
            (
                self.reservations.len(),
                MAX_VA_RESERVATIONS_V1,
                MemoryRecordKindV1::VaReservation,
            ),
            (
                self.allocations.len(),
                MAX_MEMORY_ALLOCATIONS_V1,
                MemoryRecordKindV1::Allocation,
            ),
            (
                self.mappings.len(),
                MAX_MEMORY_MAPPINGS_V1,
                MemoryRecordKindV1::Mapping,
            ),
            (
                self.publications.len(),
                MAX_MEMORY_PUBLICATIONS_V1,
                MemoryRecordKindV1::Publication,
            ),
        ] {
            if actual > maximum {
                return Err(MemoryInvariantViolationV1::CapacityExceeded(kind));
            }
        }
        Ok(())
    }

    fn validate_vms(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (index, vm) in self.vms.iter().enumerate() {
            let key = vm.admission.model_key();
            let reference = MemoryRecordRefV1::Vm(key);
            if vm.admission.domain_id() != self.domain_id
                || vm
                    .mapping_devices
                    .iter()
                    .any(|device| device.domain_id() != self.domain_id)
            {
                return Err(MemoryInvariantViolationV1::DomainMismatch(reference));
            }
            if key.id.0 == 0 || vm.handle.0 == 0 {
                return Err(MemoryInvariantViolationV1::InvalidIdentity(reference));
            }
            if !valid_mapping_admissions(&vm.mapping_devices, key.device) {
                return Err(MemoryInvariantViolationV1::DeviceSetMismatch(reference));
            }
            if !valid_page_range(vm.aperture) {
                return Err(MemoryInvariantViolationV1::InvalidRange(reference));
            }
            for old in &self.vms[..index] {
                if old.admission.model_key() == key {
                    return Err(MemoryInvariantViolationV1::Duplicate(reference));
                }
                if vm.state == MemoryVmStateV1::Active
                    && old.state == MemoryVmStateV1::Active
                    && old.handle == vm.handle
                {
                    return Err(MemoryInvariantViolationV1::HandleCollision(reference));
                }
            }
        }
        Ok(())
    }

    fn validate_reservations(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (index, reservation) in self.reservations.iter().enumerate() {
            let reference = MemoryRecordRefV1::VaReservation(reservation.key);
            let Some(vm) = self.vm_opt(reservation.key.vm) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            if reservation.key.id.0 == 0 {
                return Err(MemoryInvariantViolationV1::InvalidIdentity(reference));
            }
            if !valid_alignment(reservation.alignment)
                || !reservation.range.base.is_multiple_of(reservation.alignment)
            {
                return Err(MemoryInvariantViolationV1::InvalidAlignment(reference));
            }
            if !valid_page_range(reservation.range) || !range_within(reservation.range, vm.aperture)
            {
                return Err(MemoryInvariantViolationV1::InvalidRange(reference));
            }
            if reservation.state == VaReservationStateV1::Reserved
                && vm.state != MemoryVmStateV1::Active
            {
                return Err(MemoryInvariantViolationV1::InvalidState(reference));
            }
            for old in &self.reservations[..index] {
                if old.key == reservation.key {
                    return Err(MemoryInvariantViolationV1::Duplicate(reference));
                }
                if old.key.vm == reservation.key.vm
                    && old.state == VaReservationStateV1::Reserved
                    && reservation.state == VaReservationStateV1::Reserved
                    && ranges_overlap(old.range, reservation.range)
                {
                    return Err(MemoryInvariantViolationV1::AddressOverlap(
                        old.key,
                        reservation.key,
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_allocations(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (index, allocation) in self.allocations.iter().enumerate() {
            let reference = MemoryRecordRefV1::Allocation(allocation.key);
            let Some(vm) = self.vm_opt(allocation.key.vm) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            let Some(reservation) = self.reservation_opt(allocation.reservation) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            if allocation.key.vm != allocation.reservation.vm {
                return Err(MemoryInvariantViolationV1::BindingMismatch(reference));
            }
            if allocation.key.id.0 == 0
                || allocation.key.generation.0 == 0
                || allocation.handle.0 == 0
            {
                return Err(MemoryInvariantViolationV1::InvalidIdentity(reference));
            }
            if !valid_allocation_spec(allocation.spec, *reservation) {
                return Err(MemoryInvariantViolationV1::InvalidRange(reference));
            }
            if allocation.state == MemoryAllocationStateV1::Live
                && (vm.state != MemoryVmStateV1::Active
                    || reservation.state != VaReservationStateV1::Reserved)
            {
                return Err(MemoryInvariantViolationV1::InvalidState(reference));
            }
            for old in &self.allocations[..index] {
                if old.key == allocation.key {
                    return Err(MemoryInvariantViolationV1::Duplicate(reference));
                }
                if old.key.vm == allocation.key.vm && old.key.id == allocation.key.id {
                    if old.key.generation >= allocation.key.generation {
                        return Err(MemoryInvariantViolationV1::StaleGeneration(allocation.key));
                    }
                    if old.state != MemoryAllocationStateV1::Released {
                        return Err(MemoryInvariantViolationV1::EarlyRelease(reference));
                    }
                }
                if allocation.state == MemoryAllocationStateV1::Live
                    && old.state == MemoryAllocationStateV1::Live
                    && (old.handle == allocation.handle
                        || old.reservation == allocation.reservation)
                {
                    return Err(MemoryInvariantViolationV1::HandleCollision(reference));
                }
            }
        }
        Ok(())
    }

    fn validate_mappings(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (index, mapping) in self.mappings.iter().enumerate() {
            let reference = MemoryRecordRefV1::Mapping(mapping.key);
            let Some(allocation) = self.allocation_opt(mapping.key.allocation) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            let Some(vm) = self.vm_opt(mapping.key.allocation.vm) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            if mapping.key.id.0 == 0 {
                return Err(MemoryInvariantViolationV1::InvalidIdentity(reference));
            }
            if !device_keys_equal_admissions(&mapping.target_devices, &vm.mapping_devices) {
                return Err(MemoryInvariantViolationV1::DeviceSetMismatch(reference));
            }
            if mapping.mapped_start > mapping.mapped_end
                || mapping.mapped_end > mapping.target_devices.len()
            {
                return Err(MemoryInvariantViolationV1::InvalidRange(reference));
            }
            let range_is_valid = match mapping.state {
                MemoryMappingStateV1::MapPending => {
                    mapping.mapped_start == 0 && mapping.mapped_end == 0
                }
                MemoryMappingStateV1::MapFailed => mapping.mapped_start == 0,
                MemoryMappingStateV1::Mapped => {
                    mapping.mapped_start == 0 && mapping.mapped_end == mapping.target_devices.len()
                }
                MemoryMappingStateV1::UnmapPending => mapping.mapped_start < mapping.mapped_end,
                MemoryMappingStateV1::UnmapFailed | MemoryMappingStateV1::Ambiguous => true,
                MemoryMappingStateV1::Unmapped => mapping.mapped_start == mapping.mapped_end,
                MemoryMappingStateV1::Released => mapping.mapped_start == mapping.mapped_end,
            };
            if !range_is_valid {
                return Err(MemoryInvariantViolationV1::InvalidState(reference));
            }
            if mapping.state != MemoryMappingStateV1::Released
                && allocation.state != MemoryAllocationStateV1::Live
            {
                return Err(MemoryInvariantViolationV1::EarlyRelease(reference));
            }
            if self.mappings[..index]
                .iter()
                .any(|old| old.key == mapping.key)
            {
                return Err(MemoryInvariantViolationV1::Duplicate(reference));
            }
        }
        Ok(())
    }

    fn validate_publications(&self) -> Result<(), MemoryInvariantViolationV1> {
        for (index, publication) in self.publications.iter().enumerate() {
            let reference = MemoryRecordRefV1::Publication(publication.key);
            let Some(mapping) = self.mapping_opt(publication.key.mapping) else {
                return Err(MemoryInvariantViolationV1::MissingParent(reference));
            };
            if publication.key.id.0 == 0 {
                return Err(MemoryInvariantViolationV1::InvalidIdentity(reference));
            }
            if let MemoryPublicationOwnerV1::ComputeAqlQueue(queue) = publication.owner
                && (queue.vm != publication.key.mapping.allocation.vm
                    || queue.id.0 == 0
                    || queue.generation.0 == 0)
            {
                return Err(MemoryInvariantViolationV1::BindingMismatch(reference));
            }
            if publication.state == MemoryPublicationStateV1::Live
                && mapping.state != MemoryMappingStateV1::Mapped
            {
                return Err(MemoryInvariantViolationV1::EarlyRelease(reference));
            }
            if self.publications[..index]
                .iter()
                .any(|old| old.key == publication.key)
            {
                return Err(MemoryInvariantViolationV1::Duplicate(reference));
            }
        }
        Ok(())
    }

    fn validate_release_order(&self) -> Result<(), MemoryInvariantViolationV1> {
        for allocation in &self.allocations {
            if allocation.state == MemoryAllocationStateV1::Released
                && (self.mappings.iter().any(|mapping| {
                    mapping.key.allocation == allocation.key
                        && mapping.state != MemoryMappingStateV1::Released
                }) || self.publications.iter().any(|publication| {
                    publication.key.mapping.allocation == allocation.key
                        && publication.state != MemoryPublicationStateV1::Released
                }))
            {
                return Err(MemoryInvariantViolationV1::EarlyRelease(
                    MemoryRecordRefV1::Allocation(allocation.key),
                ));
            }
        }
        for reservation in &self.reservations {
            if reservation.state == VaReservationStateV1::Released
                && self.allocations.iter().any(|allocation| {
                    allocation.reservation == reservation.key
                        && allocation.state != MemoryAllocationStateV1::Released
                })
            {
                return Err(MemoryInvariantViolationV1::EarlyRelease(
                    MemoryRecordRefV1::VaReservation(reservation.key),
                ));
            }
        }
        for vm in &self.vms {
            let key = vm.admission.model_key();
            if vm.state == MemoryVmStateV1::Retired
                && (self.reservations.iter().any(|record| {
                    record.key.vm == key && record.state != VaReservationStateV1::Released
                }) || self.allocations.iter().any(|record| {
                    record.key.vm == key && record.state != MemoryAllocationStateV1::Released
                }) || self.mappings.iter().any(|record| {
                    record.key.allocation.vm == key
                        && record.state != MemoryMappingStateV1::Released
                }) || self.publications.iter().any(|record| {
                    record.key.mapping.allocation.vm == key
                        && record.state != MemoryPublicationStateV1::Released
                }))
            {
                return Err(MemoryInvariantViolationV1::EarlyRelease(
                    MemoryRecordRefV1::Vm(key),
                ));
            }
        }
        Ok(())
    }

    fn require_vm_active(&self, key: VmKeyV1) -> Result<(), MemoryTransitionErrorV1> {
        if self.vm(key)?.state != MemoryVmStateV1::Active {
            return Err(MemoryTransitionErrorV1::IllegalState(
                MemoryRecordRefV1::Vm(key),
            ));
        }
        Ok(())
    }

    fn vm_opt(&self, key: VmKeyV1) -> Option<&MemoryVmRecordV1> {
        self.vms
            .iter()
            .find(|record| record.admission.model_key() == key)
    }

    fn vm(&self, key: VmKeyV1) -> Result<&MemoryVmRecordV1, MemoryTransitionErrorV1> {
        self.vm_opt(key)
            .ok_or(MemoryTransitionErrorV1::NotFound(MemoryRecordRefV1::Vm(
                key,
            )))
    }

    fn vm_mut(&mut self, key: VmKeyV1) -> Result<&mut MemoryVmRecordV1, MemoryTransitionErrorV1> {
        self.vms
            .iter_mut()
            .find(|record| record.admission.model_key() == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(MemoryRecordRefV1::Vm(
                key,
            )))
    }

    fn reservation_opt(&self, key: VaReservationKeyV1) -> Option<&VaReservationRecordV1> {
        self.reservations.iter().find(|record| record.key == key)
    }

    fn reservation(
        &self,
        key: VaReservationKeyV1,
    ) -> Result<&VaReservationRecordV1, MemoryTransitionErrorV1> {
        self.reservation_opt(key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::VaReservation(key),
            ))
    }

    fn reservation_mut(
        &mut self,
        key: VaReservationKeyV1,
    ) -> Result<&mut VaReservationRecordV1, MemoryTransitionErrorV1> {
        self.reservations
            .iter_mut()
            .find(|record| record.key == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::VaReservation(key),
            ))
    }

    fn allocation_opt(&self, key: MemoryAllocationKeyV1) -> Option<&MemoryAllocationRecordV1> {
        self.allocations.iter().find(|record| record.key == key)
    }

    fn allocation(
        &self,
        key: MemoryAllocationKeyV1,
    ) -> Result<&MemoryAllocationRecordV1, MemoryTransitionErrorV1> {
        self.allocation_opt(key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Allocation(key),
            ))
    }

    fn allocation_mut(
        &mut self,
        key: MemoryAllocationKeyV1,
    ) -> Result<&mut MemoryAllocationRecordV1, MemoryTransitionErrorV1> {
        self.allocations
            .iter_mut()
            .find(|record| record.key == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Allocation(key),
            ))
    }

    fn mapping_opt(&self, key: MemoryMappingKeyV1) -> Option<&MemoryMappingRecordV1> {
        self.mappings.iter().find(|record| record.key == key)
    }

    fn mapping(
        &self,
        key: MemoryMappingKeyV1,
    ) -> Result<&MemoryMappingRecordV1, MemoryTransitionErrorV1> {
        self.mapping_opt(key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Mapping(key),
            ))
    }

    fn mapping_mut(
        &mut self,
        key: MemoryMappingKeyV1,
    ) -> Result<&mut MemoryMappingRecordV1, MemoryTransitionErrorV1> {
        self.mappings
            .iter_mut()
            .find(|record| record.key == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Mapping(key),
            ))
    }

    fn publication_mut(
        &mut self,
        key: MemoryPublicationKeyV1,
    ) -> Result<&mut MemoryPublicationRecordV1, MemoryTransitionErrorV1> {
        self.publications
            .iter_mut()
            .find(|record| record.key == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Publication(key),
            ))
    }

    fn publication(
        &self,
        key: MemoryPublicationKeyV1,
    ) -> Result<&MemoryPublicationRecordV1, MemoryTransitionErrorV1> {
        self.publications
            .iter()
            .find(|record| record.key == key)
            .ok_or(MemoryTransitionErrorV1::NotFound(
                MemoryRecordRefV1::Publication(key),
            ))
    }
}

fn ensure_memory_room(
    actual: usize,
    maximum: usize,
    kind: MemoryRecordKindV1,
) -> Result<(), MemoryTransitionErrorV1> {
    if actual >= maximum {
        return Err(MemoryTransitionErrorV1::CapacityExceeded { kind, maximum });
    }
    Ok(())
}

fn valid_alignment(alignment: u64) -> bool {
    alignment >= MEMORY_PAGE_BYTES_V1 && alignment.is_power_of_two()
}

fn valid_page_range(range: GpuVaRangeV1) -> bool {
    range.byte_len != 0
        && range.base.is_multiple_of(MEMORY_PAGE_BYTES_V1)
        && range.byte_len.is_multiple_of(MEMORY_PAGE_BYTES_V1)
        && range.checked_end().is_some()
}

fn range_within(inner: GpuVaRangeV1, outer: GpuVaRangeV1) -> bool {
    match (inner.checked_end(), outer.checked_end()) {
        (Some(inner_end), Some(outer_end)) => inner.base >= outer.base && inner_end <= outer_end,
        _ => false,
    }
}

fn ranges_overlap(left: GpuVaRangeV1, right: GpuVaRangeV1) -> bool {
    match (left.checked_end(), right.checked_end()) {
        (Some(left_end), Some(right_end)) => left.base < right_end && right.base < left_end,
        _ => true,
    }
}

fn valid_mapping_admissions(admissions: &[ModelDeviceAdmissionV1], primary: DeviceKeyV1) -> bool {
    !admissions.is_empty()
        && admissions.len() <= MAX_MEMORY_MAPPING_DEVICES_V1
        && admissions
            .windows(2)
            .all(|pair| pair[0].model_key() < pair[1].model_key())
        && admissions.iter().enumerate().all(|(index, admission)| {
            admissions[..index]
                .iter()
                .all(|old| old.model_key().physical != admission.model_key().physical)
        })
        && admissions
            .iter()
            .any(|admission| admission.model_key() == primary)
}

fn device_keys_equal_admissions(
    keys: &[DeviceKeyV1],
    admissions: &[ModelDeviceAdmissionV1],
) -> bool {
    keys.len() == admissions.len()
        && keys
            .iter()
            .zip(admissions)
            .all(|(key, admission)| *key == admission.model_key())
}

fn valid_allocation_spec(spec: MemoryAllocationSpecV1, reservation: VaReservationRecordV1) -> bool {
    spec.byte_len != 0
        && spec.byte_len <= reservation.range.byte_len
        && spec.byte_len.is_multiple_of(MEMORY_PAGE_BYTES_V1)
        && valid_alignment(spec.alignment)
        && reservation.range.base.is_multiple_of(spec.alignment)
        && spec.alignment <= reservation.alignment
}
