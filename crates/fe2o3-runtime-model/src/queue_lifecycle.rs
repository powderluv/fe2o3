//! Bounded, syscall-free compute-AQL queue lifecycle model.
//!
//! This module models queue ownership and failure policy only. It deliberately
//! does not model target resource sizes, doorbell arithmetic, packet
//! publication, signals, completion, or firmware behavior.

use alloc::vec::Vec;

use crate::*;

pub const QUEUE_LIFECYCLE_SCHEMA_VERSION_V1: u16 = 1;
pub const COMPUTE_AQL_RESOURCE_COUNT_V1: usize = 4;
pub const MAX_COMPUTE_AQL_QUEUES_V1: usize = 16;
pub const MAX_QUEUE_HISTORY_ENTRIES_V1: usize = 256;
pub const CREATE_QUEUE_ID_SENTINEL_V1: u32 = u32::MAX;

/// The single target profile admitted by this foundation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeAqlTargetProfileV1 {
    Gfx942XnackMinusSpxNps1Kfd1_18,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeAqlResourceRoleV1 {
    Ring,
    Control,
    EndOfPipe,
    ContextSave,
}

/// One exact mapped resource and the memory publication that retains it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeAqlResourceBindingV1 {
    pub mapping: MemoryMappingKeyV1,
    pub publication: MemoryPublicationKeyV1,
    /// Policy supplied by the reviewed target adapter, not a hardware fact
    /// inferred by this model.
    pub expected_kind: MemoryKindV1,
    pub expected_coherence: MemoryCoherenceV1,
    pub expected_access: MemoryAccessV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeAqlQueueResourcesV1 {
    pub ring: ComputeAqlResourceBindingV1,
    pub control: ComputeAqlResourceBindingV1,
    pub eop: ComputeAqlResourceBindingV1,
    pub context_save: ComputeAqlResourceBindingV1,
}

impl ComputeAqlQueueResourcesV1 {
    pub const fn ordered(self) -> [(ComputeAqlResourceRoleV1, ComputeAqlResourceBindingV1); 4] {
        [
            (ComputeAqlResourceRoleV1::Ring, self.ring),
            (ComputeAqlResourceRoleV1::Control, self.control),
            (ComputeAqlResourceRoleV1::EndOfPipe, self.eop),
            (ComputeAqlResourceRoleV1::ContextSave, self.context_save),
        ]
    }

    pub fn contains_mapping(self, mapping: MemoryMappingKeyV1) -> bool {
        self.ordered()
            .iter()
            .any(|(_, binding)| binding.mapping == mapping)
    }

    fn shares_mapping_with(self, other: Self) -> bool {
        self.ordered().iter().any(|(_, left)| {
            other
                .ordered()
                .iter()
                .any(|(_, right)| left.mapping == right.mapping)
        })
    }
}

/// Exact model-only plan for one compute-AQL queue incarnation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeAqlQueuePlanV1 {
    pub schema_version: u16,
    pub target: ComputeAqlTargetProfileV1,
    pub domain_id: DeviceObservationDomainIdV1,
    pub plan_id: QueuePlanIdV1,
    pub current_device: ModelDeviceAdmissionV1,
    pub queue: QueueKeyV1,
    pub initial_configuration: QueueConfigurationIdV1,
    pub resources: ComputeAqlQueueResourcesV1,
}

/// Process-local KFD queue ID observation. It is not durable identity or
/// evidence that a queue exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct UntrustedQueueIdObservationV1(pub u32);

/// Model-only classification of the CREATE_QUEUE output field relative to the
/// exact sentinel written before the ioctl. Returned IDs remain opaque and
/// include zero. A concrete adapter must separately admit the returned value
/// against its authenticated source profile before constructing this input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreateQueueIdFieldObservationV1 {
    SentinelUnchanged,
    Returned(UntrustedQueueIdObservationV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueSyscallStatusV1 {
    Succeeded,
    FailedNoEffect,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueCreateObservationV1 {
    pub status: QueueSyscallStatusV1,
    pub queue_id_field: CreateQueueIdFieldObservationV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeAqlQueuePhaseV1 {
    Planned,
    CreatePending,
    Active,
    UpdatePending,
    DisablePending,
    Disabled,
    DestroyPending,
    CancelledBeforeCreate,
    Destroyed,
    Ambiguous,
}

impl ComputeAqlQueuePhaseV1 {
    pub const fn retains_resources(self) -> bool {
        !matches!(self, Self::CancelledBeforeCreate | Self::Destroyed)
    }

    pub const fn may_have_native_queue(self) -> bool {
        matches!(
            self,
            Self::CreatePending
                | Self::Active
                | Self::UpdatePending
                | Self::DisablePending
                | Self::Disabled
                | Self::DestroyPending
                | Self::Ambiguous
        )
    }

    const fn has_exclusive_native_queue_id(self) -> bool {
        matches!(
            self,
            Self::Active
                | Self::UpdatePending
                | Self::DisablePending
                | Self::Disabled
                | Self::DestroyPending
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueHistoryEventKindV1 {
    PlanAdmitted,
    PlanCancelled,
    CreateBegan,
    CreateSucceeded,
    CreateFailedNoEffect,
    CreateAmbiguous,
    UpdateBegan,
    UpdateSucceeded,
    UpdateFailedNoEffect,
    UpdateAmbiguous,
    DisableBegan,
    DisableSucceeded,
    DisableFailedNoEffect,
    DisableAmbiguous,
    DestroyBegan,
    DestroySucceeded,
    DestroyFailedNoEffect,
    DestroyAmbiguous,
    CurrentnessLost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueHistoryEntryV1 {
    pub sequence: u64,
    pub queue: QueueKeyV1,
    pub event: QueueHistoryEventKindV1,
    pub from: ComputeAqlQueuePhaseV1,
    pub to: ComputeAqlQueuePhaseV1,
    pub queue_id: Option<UntrustedQueueIdObservationV1>,
    pub configuration: QueueConfigurationIdV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeAqlQueueRecordV1 {
    pub plan: ComputeAqlQueuePlanV1,
    pub phase: ComputeAqlQueuePhaseV1,
    pub queue_id: Option<UntrustedQueueIdObservationV1>,
    pub configuration: QueueConfigurationIdV1,
    pub pending_configuration: Option<QueueConfigurationIdV1>,
    resume_phase: Option<ComputeAqlQueuePhaseV1>,
}

impl ComputeAqlQueueRecordV1 {
    fn reserves_known_queue_id(self, candidate: UntrustedQueueIdObservationV1) -> bool {
        (self.phase.has_exclusive_native_queue_id()
            || self.phase == ComputeAqlQueuePhaseV1::Ambiguous)
            && self.queue_id == Some(candidate)
    }

    fn has_unresolved_possibly_native_identity(self) -> bool {
        matches!(
            self.phase,
            ComputeAqlQueuePhaseV1::CreatePending | ComputeAqlQueuePhaseV1::Ambiguous
        ) && self.queue_id.is_none()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueTransitionV1 {
    CancelPlan {
        queue: QueueKeyV1,
    },
    BeginCreate {
        queue: QueueKeyV1,
    },
    ObserveCreate {
        queue: QueueKeyV1,
        observation: QueueCreateObservationV1,
    },
    BeginUpdate {
        queue: QueueKeyV1,
        configuration: QueueConfigurationIdV1,
    },
    ObserveUpdate {
        queue: QueueKeyV1,
        status: QueueSyscallStatusV1,
    },
    BeginDisable {
        queue: QueueKeyV1,
    },
    ObserveDisable {
        queue: QueueKeyV1,
        status: QueueSyscallStatusV1,
    },
    BeginDestroy {
        queue: QueueKeyV1,
    },
    ObserveDestroy {
        queue: QueueKeyV1,
        status: QueueSyscallStatusV1,
    },
    ObserveCurrentnessLost {
        queue: QueueKeyV1,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueRecordKindV1 {
    Queue,
    History,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueInvariantViolationV1 {
    CapacityExceeded(QueueRecordKindV1),
    DomainMismatch(QueueKeyV1),
    DuplicateQueue(QueueKeyV1),
    InvalidIdentity(QueueKeyV1),
    StaleQueueGeneration(QueueKeyV1),
    DeviceNotCurrent(QueueKeyV1),
    VmNotCurrent(QueueKeyV1),
    MissingResource(QueueKeyV1, ComputeAqlResourceRoleV1),
    ResourceBindingMismatch(QueueKeyV1, ComputeAqlResourceRoleV1),
    ResourceAlias(QueueKeyV1),
    ResourceReleasedEarly(QueueKeyV1, ComputeAqlResourceRoleV1),
    QueueIdCollision(QueueKeyV1),
    ConcurrentCreatePending(QueueKeyV1),
    InvalidPhase(QueueKeyV1),
    InvalidHistory(QueueKeyV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueueTransitionErrorV1 {
    SourceInvariant(QueueInvariantViolationV1),
    NextInvariant(QueueInvariantViolationV1),
    Memory(MemoryTransitionErrorV1),
    CapacityExceeded {
        kind: QueueRecordKindV1,
        maximum: usize,
    },
    NotFound(QueueKeyV1),
    AlreadyExists(QueueKeyV1),
    InvalidPlan(QueueInvariantViolationV1),
    IllegalState {
        queue: QueueKeyV1,
        phase: ComputeAqlQueuePhaseV1,
    },
    InvalidConfiguration(QueueKeyV1),
    QueueCreationPoisoned(QueueKeyV1),
    ResourceInUse(MemoryMappingKeyV1),
}

/// Append-only queue records and transition history for one observation domain.
///
/// Values in this state are freely constructible model inputs. They grant no
/// file descriptor, mapping, queue, doorbell, dispatch, or proof authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueLifecycleStateV1 {
    domain_id: DeviceObservationDomainIdV1,
    queues: Vec<ComputeAqlQueueRecordV1>,
    history: Vec<QueueHistoryEntryV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuePlanAdmissionV1 {
    queue_state: QueueLifecycleStateV1,
    memory_state: MemoryLifecycleStateV1,
}

impl QueuePlanAdmissionV1 {
    pub const fn authority_domain(&self) -> AuthorityDomainV1 {
        AuthorityDomainV1::ModelOnly
    }

    pub const fn queue_state(&self) -> &QueueLifecycleStateV1 {
        &self.queue_state
    }

    pub const fn memory_state(&self) -> &MemoryLifecycleStateV1 {
        &self.memory_state
    }

    pub fn into_states(self) -> (QueueLifecycleStateV1, MemoryLifecycleStateV1) {
        (self.queue_state, self.memory_state)
    }
}

impl QueueLifecycleStateV1 {
    pub const fn new(domain_id: DeviceObservationDomainIdV1) -> Self {
        Self {
            domain_id,
            queues: Vec::new(),
            history: Vec::new(),
        }
    }

    pub const fn authority_domain(&self) -> AuthorityDomainV1 {
        AuthorityDomainV1::ModelOnly
    }

    pub const fn domain_id(&self) -> DeviceObservationDomainIdV1 {
        self.domain_id
    }

    pub fn queues(&self) -> &[ComputeAqlQueueRecordV1] {
        &self.queues
    }

    pub fn history(&self) -> &[QueueHistoryEntryV1] {
        &self.history
    }

    pub fn admit_compute_aql_plan(
        &self,
        identity: &DeviceIdentityStateV1,
        memory: &MemoryLifecycleStateV1,
        plan: ComputeAqlQueuePlanV1,
    ) -> Result<QueuePlanAdmissionV1, QueueTransitionErrorV1> {
        self.validate_global_invariants(identity, memory)
            .map_err(QueueTransitionErrorV1::SourceInvariant)?;
        ensure_queue_room(
            self.queues.len(),
            MAX_COMPUTE_AQL_QUEUES_V1,
            QueueRecordKindV1::Queue,
        )?;
        ensure_queue_room(
            self.history.len(),
            MAX_QUEUE_HISTORY_ENTRIES_V1,
            QueueRecordKindV1::History,
        )?;
        self.validate_new_plan(identity, memory, plan)
            .map_err(QueueTransitionErrorV1::InvalidPlan)?;

        let mut next_memory = memory.clone();
        for (_, resource) in plan.resources.ordered() {
            next_memory = next_memory
                .publish_compute_aql_queue_mapping(resource.publication, plan.queue)
                .map_err(QueueTransitionErrorV1::Memory)?;
        }

        let mut next = self.clone();
        let record = ComputeAqlQueueRecordV1 {
            plan,
            phase: ComputeAqlQueuePhaseV1::Planned,
            queue_id: None,
            configuration: plan.initial_configuration,
            pending_configuration: None,
            resume_phase: None,
        };
        next.queues.push(record);
        next.push_history(
            record,
            QueueHistoryEventKindV1::PlanAdmitted,
            ComputeAqlQueuePhaseV1::Planned,
        )?;
        next.validate_global_invariants(identity, &next_memory)
            .map_err(QueueTransitionErrorV1::NextInvariant)?;
        Ok(QueuePlanAdmissionV1 {
            queue_state: next,
            memory_state: next_memory,
        })
    }

    pub fn next(
        &self,
        identity: &DeviceIdentityStateV1,
        memory: &MemoryLifecycleStateV1,
        transition: QueueTransitionV1,
    ) -> Result<Self, QueueTransitionErrorV1> {
        self.validate_global_invariants(identity, memory)
            .map_err(QueueTransitionErrorV1::SourceInvariant)?;
        ensure_queue_room(
            self.history.len(),
            MAX_QUEUE_HISTORY_ENTRIES_V1,
            QueueRecordKindV1::History,
        )?;
        let mut next = self.clone();
        next.apply(transition)?;
        next.validate_global_invariants(identity, memory)
            .map_err(QueueTransitionErrorV1::NextInvariant)?;
        Ok(next)
    }

    /// Records loss of the external currentness contract without requiring a
    /// now-stale identity or memory snapshot to validate as current.
    pub fn quarantine_currentness_loss(
        &self,
        queue: QueueKeyV1,
    ) -> Result<Self, QueueTransitionErrorV1> {
        ensure_queue_room(
            self.history.len(),
            MAX_QUEUE_HISTORY_ENTRIES_V1,
            QueueRecordKindV1::History,
        )?;
        let mut next = self.clone();
        next.apply(QueueTransitionV1::ObserveCurrentnessLost { queue })?;
        next.validate_internal_invariants()
            .map_err(QueueTransitionErrorV1::NextInvariant)?;
        Ok(next)
    }

    /// Releases the memory publications only after the queue can no longer
    /// exist. The returned memory state remains model-only.
    pub fn release_resource_publications(
        &self,
        memory: &MemoryLifecycleStateV1,
        queue: QueueKeyV1,
    ) -> Result<MemoryLifecycleStateV1, QueueTransitionErrorV1> {
        let record = self.queue(queue)?;
        if record.phase.retains_resources() {
            return Err(QueueTransitionErrorV1::IllegalState {
                queue,
                phase: record.phase,
            });
        }
        let mut next = memory.clone();
        for (_, resource) in record.plan.resources.ordered() {
            next = next
                .release_compute_aql_queue_publication(resource.publication, queue)
                .map_err(QueueTransitionErrorV1::Memory)?;
        }
        Ok(next)
    }

    pub fn can_release_mapping(&self, mapping: MemoryMappingKeyV1) -> bool {
        !self.queues.iter().any(|record| {
            record.phase.retains_resources() && record.plan.resources.contains_mapping(mapping)
        })
    }

    pub fn validate_global_invariants(
        &self,
        identity: &DeviceIdentityStateV1,
        memory: &MemoryLifecycleStateV1,
    ) -> Result<(), QueueInvariantViolationV1> {
        self.validate_internal_invariants()?;
        for record in &self.queues {
            if record.phase.retains_resources() {
                self.validate_context(identity, memory, record, true)?;
            }
        }
        Ok(())
    }

    fn validate_internal_invariants(&self) -> Result<(), QueueInvariantViolationV1> {
        if self.queues.len() > MAX_COMPUTE_AQL_QUEUES_V1 {
            return Err(QueueInvariantViolationV1::CapacityExceeded(
                QueueRecordKindV1::Queue,
            ));
        }
        if self.history.len() > MAX_QUEUE_HISTORY_ENTRIES_V1 {
            return Err(QueueInvariantViolationV1::CapacityExceeded(
                QueueRecordKindV1::History,
            ));
        }
        for (index, record) in self.queues.iter().enumerate() {
            self.validate_plan_shape(record.plan)?;
            if self.queues[..index]
                .iter()
                .any(|other| other.plan.queue == record.plan.queue)
            {
                return Err(QueueInvariantViolationV1::DuplicateQueue(record.plan.queue));
            }
            for older in self.queues[..index].iter().filter(|older| {
                older.plan.queue.vm == record.plan.queue.vm
                    && older.plan.queue.id == record.plan.queue.id
            }) {
                if older.plan.queue.generation >= record.plan.queue.generation
                    || older.phase.retains_resources()
                {
                    return Err(QueueInvariantViolationV1::StaleQueueGeneration(
                        record.plan.queue,
                    ));
                }
            }
            self.validate_phase(record)?;
            if record.phase.retains_resources()
                && self.queues[..index].iter().any(|other| {
                    other.phase.retains_resources()
                        && other
                            .plan
                            .resources
                            .shares_mapping_with(record.plan.resources)
                })
            {
                return Err(QueueInvariantViolationV1::ResourceAlias(record.plan.queue));
            }
            if record.phase == ComputeAqlQueuePhaseV1::CreatePending
                && self.queues[..index]
                    .iter()
                    .any(|other| other.phase == ComputeAqlQueuePhaseV1::CreatePending)
            {
                return Err(QueueInvariantViolationV1::ConcurrentCreatePending(
                    record.plan.queue,
                ));
            }
            if let Some(queue_id) = record.queue_id
                && (record.phase.has_exclusive_native_queue_id()
                    || record.phase == ComputeAqlQueuePhaseV1::Ambiguous)
                && self.queues[..index]
                    .iter()
                    .any(|other| other.reserves_known_queue_id(queue_id))
            {
                return Err(QueueInvariantViolationV1::QueueIdCollision(
                    record.plan.queue,
                ));
            }
        }
        self.validate_history()
    }

    fn validate_new_plan(
        &self,
        identity: &DeviceIdentityStateV1,
        memory: &MemoryLifecycleStateV1,
        plan: ComputeAqlQueuePlanV1,
    ) -> Result<(), QueueInvariantViolationV1> {
        self.validate_plan_shape(plan)?;
        if self
            .queues
            .iter()
            .any(|record| record.plan.queue == plan.queue)
        {
            return Err(QueueInvariantViolationV1::DuplicateQueue(plan.queue));
        }
        if self.queues.iter().any(|record| {
            record.phase.retains_resources()
                && record.plan.resources.shares_mapping_with(plan.resources)
        }) {
            return Err(QueueInvariantViolationV1::ResourceAlias(plan.queue));
        }
        if let Some(newest) = self
            .queues
            .iter()
            .filter(|record| {
                record.plan.queue.vm == plan.queue.vm && record.plan.queue.id == plan.queue.id
            })
            .max_by_key(|record| record.plan.queue.generation)
            && (newest.phase.retains_resources()
                || newest
                    .plan
                    .queue
                    .generation
                    .0
                    .checked_add(1)
                    .is_none_or(|next| next != plan.queue.generation.0))
        {
            return Err(QueueInvariantViolationV1::StaleQueueGeneration(plan.queue));
        }
        let record = ComputeAqlQueueRecordV1 {
            plan,
            phase: ComputeAqlQueuePhaseV1::Planned,
            queue_id: None,
            configuration: plan.initial_configuration,
            pending_configuration: None,
            resume_phase: None,
        };
        self.validate_context(identity, memory, &record, false)
    }

    fn validate_plan_shape(
        &self,
        plan: ComputeAqlQueuePlanV1,
    ) -> Result<(), QueueInvariantViolationV1> {
        let queue = plan.queue;
        if plan.schema_version != QUEUE_LIFECYCLE_SCHEMA_VERSION_V1
            || plan.domain_id != self.domain_id
            || plan.current_device.domain_id() != self.domain_id
            || queue.vm.device != plan.current_device.model_key()
        {
            return Err(QueueInvariantViolationV1::DomainMismatch(queue));
        }
        if queue.vm.id.0 == 0
            || queue.id.0 == 0
            || queue.generation.0 == 0
            || digest_is_zero(plan.plan_id.digest())
            || digest_is_zero(plan.initial_configuration.digest())
        {
            return Err(QueueInvariantViolationV1::InvalidIdentity(queue));
        }
        let resources = plan.resources.ordered();
        for (role, resource) in resources {
            if resource.mapping.allocation.vm != queue.vm
                || resource.mapping.id.0 == 0
                || resource.publication.mapping != resource.mapping
                || resource.publication.id.0 == 0
            {
                return Err(QueueInvariantViolationV1::ResourceBindingMismatch(
                    queue, role,
                ));
            }
        }
        for left in 0..resources.len() {
            for right in (left + 1)..resources.len() {
                if resources[left].1.mapping == resources[right].1.mapping
                    || resources[left].1.mapping.allocation == resources[right].1.mapping.allocation
                    || resources[left].1.publication == resources[right].1.publication
                {
                    return Err(QueueInvariantViolationV1::ResourceAlias(queue));
                }
            }
        }
        Ok(())
    }

    fn validate_context(
        &self,
        identity: &DeviceIdentityStateV1,
        memory: &MemoryLifecycleStateV1,
        record: &ComputeAqlQueueRecordV1,
        publication_required: bool,
    ) -> Result<(), QueueInvariantViolationV1> {
        let plan = record.plan;
        let queue = plan.queue;
        if identity.domain_id() != self.domain_id || memory.domain_id() != self.domain_id {
            return Err(QueueInvariantViolationV1::DomainMismatch(queue));
        }
        let current = identity
            .devices()
            .iter()
            .find(|candidate| candidate.key == queue.vm.device)
            .filter(|candidate| {
                candidate.status == ModelAdmissionStatusV1::Active
                    && candidate.domain_id == self.domain_id
                    && candidate.profile_id == plan.current_device.correlation().profile_id()
                    && candidate.correlation == plan.current_device.correlation()
            })
            .ok_or(QueueInvariantViolationV1::DeviceNotCurrent(queue))?;
        if current.key != plan.current_device.model_key() {
            return Err(QueueInvariantViolationV1::DeviceNotCurrent(queue));
        }
        if !identity.vms().iter().any(|candidate| {
            candidate.key == queue.vm
                && candidate.domain_id == self.domain_id
                && candidate.status == ModelAdmissionStatusV1::Active
        }) {
            return Err(QueueInvariantViolationV1::VmNotCurrent(queue));
        }
        if !memory.vms().iter().any(|candidate| {
            candidate.admission.model_key() == queue.vm
                && candidate.state == MemoryVmStateV1::Active
        }) {
            return Err(QueueInvariantViolationV1::VmNotCurrent(queue));
        }
        for (role, resource) in plan.resources.ordered() {
            let mapping = memory
                .mappings()
                .iter()
                .find(|candidate| candidate.key == resource.mapping)
                .ok_or(QueueInvariantViolationV1::MissingResource(queue, role))?;
            let allocation = memory
                .allocations()
                .iter()
                .find(|candidate| candidate.key == resource.mapping.allocation)
                .ok_or(QueueInvariantViolationV1::MissingResource(queue, role))?;
            let publication = memory
                .publications()
                .iter()
                .find(|candidate| candidate.key == resource.publication);
            if mapping.state != MemoryMappingStateV1::Mapped
                || mapping.access != resource.expected_access
                || mapping.mapped_start != 0
                || mapping.mapped_end != 1
                || mapping.target_devices.as_slice() != [queue.vm.device]
                || allocation.state != MemoryAllocationStateV1::Live
                || allocation.spec.kind != resource.expected_kind
                || allocation.spec.coherence != resource.expected_coherence
            {
                return Err(QueueInvariantViolationV1::ResourceReleasedEarly(
                    queue, role,
                ));
            }
            match (publication_required, publication) {
                (true, Some(publication))
                    if publication.state == MemoryPublicationStateV1::Live
                        && publication.owner
                            == MemoryPublicationOwnerV1::ComputeAqlQueue(queue) => {}
                (false, None) => {}
                (true, None) => {
                    return Err(QueueInvariantViolationV1::MissingResource(queue, role));
                }
                _ => {
                    return Err(QueueInvariantViolationV1::ResourceBindingMismatch(
                        queue, role,
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_phase(
        &self,
        record: &ComputeAqlQueueRecordV1,
    ) -> Result<(), QueueInvariantViolationV1> {
        let queue = record.plan.queue;
        let queue_id_valid = record.queue_id.is_some();
        let valid = match record.phase {
            ComputeAqlQueuePhaseV1::Planned
            | ComputeAqlQueuePhaseV1::CreatePending
            | ComputeAqlQueuePhaseV1::CancelledBeforeCreate => {
                record.queue_id.is_none()
                    && record.pending_configuration.is_none()
                    && record.resume_phase.is_none()
            }
            ComputeAqlQueuePhaseV1::Active
            | ComputeAqlQueuePhaseV1::Disabled
            | ComputeAqlQueuePhaseV1::Destroyed => {
                queue_id_valid
                    && record.pending_configuration.is_none()
                    && record.resume_phase.is_none()
            }
            ComputeAqlQueuePhaseV1::UpdatePending => {
                queue_id_valid
                    && record.pending_configuration.is_some_and(|candidate| {
                        !digest_is_zero(candidate.digest()) && candidate != record.configuration
                    })
                    && matches!(
                        record.resume_phase,
                        Some(ComputeAqlQueuePhaseV1::Active | ComputeAqlQueuePhaseV1::Disabled)
                    )
            }
            ComputeAqlQueuePhaseV1::DisablePending => {
                queue_id_valid
                    && record.pending_configuration.is_none()
                    && record.resume_phase == Some(ComputeAqlQueuePhaseV1::Active)
            }
            ComputeAqlQueuePhaseV1::DestroyPending => {
                queue_id_valid
                    && record.pending_configuration.is_none()
                    && record.resume_phase == Some(ComputeAqlQueuePhaseV1::Disabled)
            }
            ComputeAqlQueuePhaseV1::Ambiguous => true,
        };
        if valid {
            Ok(())
        } else {
            Err(QueueInvariantViolationV1::InvalidPhase(queue))
        }
    }

    fn validate_history(&self) -> Result<(), QueueInvariantViolationV1> {
        for (index, entry) in self.history.iter().enumerate() {
            if entry.sequence != index as u64 + 1 || !history_edge_is_valid(*entry) {
                return Err(QueueInvariantViolationV1::InvalidHistory(entry.queue));
            }
            let previous = self.history[..index]
                .iter()
                .rev()
                .find(|candidate| candidate.queue == entry.queue);
            let linked = match (entry.event, previous) {
                (QueueHistoryEventKindV1::PlanAdmitted, None) => true,
                (QueueHistoryEventKindV1::PlanAdmitted, Some(_)) | (_, None) => false,
                (_, Some(previous)) => previous.to == entry.from,
            };
            if !linked {
                return Err(QueueInvariantViolationV1::InvalidHistory(entry.queue));
            }
        }
        for record in &self.queues {
            let last = self
                .history
                .iter()
                .rev()
                .find(|entry| entry.queue == record.plan.queue)
                .ok_or(QueueInvariantViolationV1::InvalidHistory(record.plan.queue))?;
            if last.to != record.phase
                || last.queue_id != record.queue_id
                || last.configuration != record.configuration
            {
                return Err(QueueInvariantViolationV1::InvalidHistory(record.plan.queue));
            }
        }
        Ok(())
    }

    fn apply(&mut self, transition: QueueTransitionV1) -> Result<(), QueueTransitionErrorV1> {
        let key = transition.queue();
        let index = self
            .queues
            .iter()
            .position(|record| record.plan.queue == key)
            .ok_or(QueueTransitionErrorV1::NotFound(key))?;
        let before = self.queues[index];
        let event = match transition {
            QueueTransitionV1::CancelPlan { .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::Planned])?;
                self.queues[index].phase = ComputeAqlQueuePhaseV1::CancelledBeforeCreate;
                QueueHistoryEventKindV1::PlanCancelled
            }
            QueueTransitionV1::BeginCreate { .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::Planned])?;
                if self.queues.iter().any(|record| {
                    record.plan.queue != key && record.has_unresolved_possibly_native_identity()
                }) {
                    return Err(QueueTransitionErrorV1::QueueCreationPoisoned(key));
                }
                self.queues[index].phase = ComputeAqlQueuePhaseV1::CreatePending;
                QueueHistoryEventKindV1::CreateBegan
            }
            QueueTransitionV1::ObserveCreate { observation, .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::CreatePending])?;
                self.observe_create(index, observation)
            }
            QueueTransitionV1::BeginUpdate { configuration, .. } => {
                require_queue_phase(
                    before,
                    &[
                        ComputeAqlQueuePhaseV1::Active,
                        ComputeAqlQueuePhaseV1::Disabled,
                    ],
                )?;
                if digest_is_zero(configuration.digest()) || configuration == before.configuration {
                    return Err(QueueTransitionErrorV1::InvalidConfiguration(key));
                }
                self.queues[index].pending_configuration = Some(configuration);
                self.queues[index].resume_phase = Some(before.phase);
                self.queues[index].phase = ComputeAqlQueuePhaseV1::UpdatePending;
                QueueHistoryEventKindV1::UpdateBegan
            }
            QueueTransitionV1::ObserveUpdate { status, .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::UpdatePending])?;
                match status {
                    QueueSyscallStatusV1::Succeeded => {
                        self.queues[index].configuration = before
                            .pending_configuration
                            .ok_or(QueueTransitionErrorV1::InvalidConfiguration(key))?;
                        self.queues[index].pending_configuration = None;
                        self.queues[index].resume_phase = None;
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Active;
                        QueueHistoryEventKindV1::UpdateSucceeded
                    }
                    QueueSyscallStatusV1::FailedNoEffect => {
                        self.queues[index].pending_configuration = None;
                        self.queues[index].phase = before
                            .resume_phase
                            .ok_or(QueueTransitionErrorV1::InvalidConfiguration(key))?;
                        self.queues[index].resume_phase = None;
                        QueueHistoryEventKindV1::UpdateFailedNoEffect
                    }
                    QueueSyscallStatusV1::Indeterminate => {
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Ambiguous;
                        QueueHistoryEventKindV1::UpdateAmbiguous
                    }
                }
            }
            QueueTransitionV1::BeginDisable { .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::Active])?;
                self.queues[index].resume_phase = Some(before.phase);
                self.queues[index].phase = ComputeAqlQueuePhaseV1::DisablePending;
                QueueHistoryEventKindV1::DisableBegan
            }
            QueueTransitionV1::ObserveDisable { status, .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::DisablePending])?;
                match status {
                    QueueSyscallStatusV1::Succeeded => {
                        self.queues[index].resume_phase = None;
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Disabled;
                        QueueHistoryEventKindV1::DisableSucceeded
                    }
                    QueueSyscallStatusV1::FailedNoEffect => {
                        self.queues[index].resume_phase = None;
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Active;
                        QueueHistoryEventKindV1::DisableFailedNoEffect
                    }
                    QueueSyscallStatusV1::Indeterminate => {
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Ambiguous;
                        QueueHistoryEventKindV1::DisableAmbiguous
                    }
                }
            }
            QueueTransitionV1::BeginDestroy { .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::Disabled])?;
                self.queues[index].resume_phase = Some(before.phase);
                self.queues[index].phase = ComputeAqlQueuePhaseV1::DestroyPending;
                QueueHistoryEventKindV1::DestroyBegan
            }
            QueueTransitionV1::ObserveDestroy { status, .. } => {
                require_queue_phase(before, &[ComputeAqlQueuePhaseV1::DestroyPending])?;
                match status {
                    QueueSyscallStatusV1::Succeeded => {
                        self.queues[index].resume_phase = None;
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Destroyed;
                        QueueHistoryEventKindV1::DestroySucceeded
                    }
                    QueueSyscallStatusV1::FailedNoEffect => {
                        self.queues[index].resume_phase = None;
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Disabled;
                        QueueHistoryEventKindV1::DestroyFailedNoEffect
                    }
                    QueueSyscallStatusV1::Indeterminate => {
                        self.queues[index].phase = ComputeAqlQueuePhaseV1::Ambiguous;
                        QueueHistoryEventKindV1::DestroyAmbiguous
                    }
                }
            }
            QueueTransitionV1::ObserveCurrentnessLost { .. } => {
                if !before.phase.retains_resources() {
                    return Err(QueueTransitionErrorV1::IllegalState {
                        queue: key,
                        phase: before.phase,
                    });
                }
                self.queues[index].phase = ComputeAqlQueuePhaseV1::Ambiguous;
                QueueHistoryEventKindV1::CurrentnessLost
            }
        };
        self.push_history(self.queues[index], event, before.phase)
    }

    fn observe_create(
        &mut self,
        index: usize,
        observation: QueueCreateObservationV1,
    ) -> QueueHistoryEventKindV1 {
        let queue_id = match observation.queue_id_field {
            CreateQueueIdFieldObservationV1::SentinelUnchanged => None,
            CreateQueueIdFieldObservationV1::Returned(queue_id)
                if queue_id.0 != CREATE_QUEUE_ID_SENTINEL_V1 =>
            {
                Some(queue_id)
            }
            CreateQueueIdFieldObservationV1::Returned(_) => None,
        };
        let unchanged_sentinel = matches!(
            observation.queue_id_field,
            CreateQueueIdFieldObservationV1::SentinelUnchanged
        );
        let collision = queue_id.is_some_and(|candidate| {
            self.queues.iter().enumerate().any(|(other_index, other)| {
                other_index != index && other.reserves_known_queue_id(candidate)
            })
        });
        let unresolved_other = self.queues.iter().enumerate().any(|(other_index, other)| {
            other_index != index && other.has_unresolved_possibly_native_identity()
        });
        match observation.status {
            QueueSyscallStatusV1::Succeeded
                if queue_id.is_some() && !collision && !unresolved_other =>
            {
                self.queues[index].queue_id = queue_id;
                self.queues[index].phase = ComputeAqlQueuePhaseV1::Active;
                QueueHistoryEventKindV1::CreateSucceeded
            }
            QueueSyscallStatusV1::FailedNoEffect if unchanged_sentinel => {
                self.queues[index].phase = ComputeAqlQueuePhaseV1::Planned;
                QueueHistoryEventKindV1::CreateFailedNoEffect
            }
            _ => {
                self.queues[index].queue_id = if collision || unresolved_other {
                    None
                } else {
                    queue_id
                };
                self.queues[index].phase = ComputeAqlQueuePhaseV1::Ambiguous;
                QueueHistoryEventKindV1::CreateAmbiguous
            }
        }
    }

    fn push_history(
        &mut self,
        record: ComputeAqlQueueRecordV1,
        event: QueueHistoryEventKindV1,
        from: ComputeAqlQueuePhaseV1,
    ) -> Result<(), QueueTransitionErrorV1> {
        ensure_queue_room(
            self.history.len(),
            MAX_QUEUE_HISTORY_ENTRIES_V1,
            QueueRecordKindV1::History,
        )?;
        let sequence = u64::try_from(self.history.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(QueueTransitionErrorV1::CapacityExceeded {
                kind: QueueRecordKindV1::History,
                maximum: MAX_QUEUE_HISTORY_ENTRIES_V1,
            })?;
        self.history.push(QueueHistoryEntryV1 {
            sequence,
            queue: record.plan.queue,
            event,
            from,
            to: record.phase,
            queue_id: record.queue_id,
            configuration: record.configuration,
        });
        Ok(())
    }

    fn queue(&self, key: QueueKeyV1) -> Result<&ComputeAqlQueueRecordV1, QueueTransitionErrorV1> {
        self.queues
            .iter()
            .find(|record| record.plan.queue == key)
            .ok_or(QueueTransitionErrorV1::NotFound(key))
    }
}

impl QueueTransitionV1 {
    const fn queue(self) -> QueueKeyV1 {
        match self {
            Self::CancelPlan { queue }
            | Self::BeginCreate { queue }
            | Self::ObserveCreate { queue, .. }
            | Self::BeginUpdate { queue, .. }
            | Self::ObserveUpdate { queue, .. }
            | Self::BeginDisable { queue }
            | Self::ObserveDisable { queue, .. }
            | Self::BeginDestroy { queue }
            | Self::ObserveDestroy { queue, .. }
            | Self::ObserveCurrentnessLost { queue } => queue,
        }
    }
}

fn digest_is_zero(digest: IdentityDigestV1) -> bool {
    *digest.as_bytes() == [0; IDENTITY_DIGEST_BYTES_V1]
}

fn require_queue_phase(
    record: ComputeAqlQueueRecordV1,
    allowed: &[ComputeAqlQueuePhaseV1],
) -> Result<(), QueueTransitionErrorV1> {
    if allowed.contains(&record.phase) {
        Ok(())
    } else {
        Err(QueueTransitionErrorV1::IllegalState {
            queue: record.plan.queue,
            phase: record.phase,
        })
    }
}

fn ensure_queue_room(
    actual: usize,
    maximum: usize,
    kind: QueueRecordKindV1,
) -> Result<(), QueueTransitionErrorV1> {
    if actual < maximum {
        Ok(())
    } else {
        Err(QueueTransitionErrorV1::CapacityExceeded { kind, maximum })
    }
}

fn history_edge_is_valid(entry: QueueHistoryEntryV1) -> bool {
    use ComputeAqlQueuePhaseV1 as Phase;
    use QueueHistoryEventKindV1 as Event;

    matches!(
        (entry.event, entry.from, entry.to),
        (Event::PlanAdmitted, Phase::Planned, Phase::Planned)
            | (
                Event::PlanCancelled,
                Phase::Planned,
                Phase::CancelledBeforeCreate
            )
            | (Event::CreateBegan, Phase::Planned, Phase::CreatePending)
            | (Event::CreateSucceeded, Phase::CreatePending, Phase::Active)
            | (
                Event::CreateFailedNoEffect,
                Phase::CreatePending,
                Phase::Planned
            )
            | (
                Event::CreateAmbiguous,
                Phase::CreatePending,
                Phase::Ambiguous
            )
            | (
                Event::UpdateBegan,
                Phase::Active | Phase::Disabled,
                Phase::UpdatePending
            )
            | (Event::UpdateSucceeded, Phase::UpdatePending, Phase::Active)
            | (
                Event::UpdateFailedNoEffect,
                Phase::UpdatePending,
                Phase::Active | Phase::Disabled
            )
            | (
                Event::UpdateAmbiguous,
                Phase::UpdatePending,
                Phase::Ambiguous
            )
            | (Event::DisableBegan, Phase::Active, Phase::DisablePending)
            | (
                Event::DisableSucceeded,
                Phase::DisablePending,
                Phase::Disabled
            )
            | (
                Event::DisableFailedNoEffect,
                Phase::DisablePending,
                Phase::Active
            )
            | (
                Event::DisableAmbiguous,
                Phase::DisablePending,
                Phase::Ambiguous
            )
            | (Event::DestroyBegan, Phase::Disabled, Phase::DestroyPending)
            | (
                Event::DestroySucceeded,
                Phase::DestroyPending,
                Phase::Destroyed
            )
            | (
                Event::DestroyFailedNoEffect,
                Phase::DestroyPending,
                Phase::Disabled
            )
            | (
                Event::DestroyAmbiguous,
                Phase::DestroyPending,
                Phase::Ambiguous
            )
            | (
                Event::CurrentnessLost,
                Phase::Planned
                    | Phase::CreatePending
                    | Phase::Active
                    | Phase::UpdatePending
                    | Phase::DisablePending
                    | Phase::Disabled
                    | Phase::DestroyPending
                    | Phase::Ambiguous,
                Phase::Ambiguous
            )
    )
}
